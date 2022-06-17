# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example to run loss program search."""
import random
import time

from absl import app
from absl import flags
from absl import logging
from brain_autorl.evolving_rl import evolution_v2
from brain_autorl.evolving_rl.custom_dqn import DQN
from brain_autorl.evolving_rl.custom_dqn import make_networks
from brain_autorl.evolving_rl.env_utils import run_train_loop
import brain_autorl.evolving_rl.graph_configs as gconfig
from brain_autorl.evolving_rl.program import build_program
from brain_autorl.evolving_rl.program import InvalidProgramError
import numpy as np
import pyglove as pg
import tensorflow as tf

flags.DEFINE_integer('seed', 1, 'Seed for various modules (tf, numpy, etc).')
flags.DEFINE_integer('max_trials', 100, 'Max number of vizier trials.')
flags.DEFINE_string('objective_metric', 'train/normalized_avg_return_last50',
                    'Objective to maximize')
flags.DEFINE_bool('use_priority', False, 'Whether to use priority replay.')
flags.DEFINE_integer('batch_size', 32, 'Batch size for dqn updates')
flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate for dqn')
flags.DEFINE_integer('epsilon_schedule', 10000,
                     'Steps at which to interpolate epsilon')
flags.DEFINE_string('wrapper_version', 'v3', 'Minigrid Obs wrapper version')
flags.DEFINE_integer('target_update_period', 100, 'Target update period.')
flags.DEFINE_integer('population_size', 300, 'Evolution population size.')
flags.DEFINE_integer('tournament_size', 25, 'Evolution tournament size.')
flags.DEFINE_float('mutation_probability', 0.95,
                   'Probability of performing a mutation.')
flags.DEFINE_string('graph_def', 'pre_graph_6_existingdqn_nofreeze',
                    'Graph definition name')

FLAGS = flags.FLAGS

graph_defs = {
    'pre_graph_6_existingdqn_nofreeze':
        gconfig.pre_graph_6_existingdqn_nofreeze,
    'pre_graph_7_existingdqn_nofreeze':
        gconfig.pre_graph_7_existingdqn_nofreeze,
    'pre_graph_8_scratch':
        gconfig.pre_graph_8_scratch,
    'pre_graph_9_existingdqn_nofreeze':
        gconfig.pre_graph_9_existingdqn_nofreeze,
    'pre_graph_10_existingdqn_nofreeze':
        gconfig.pre_graph_10_existingdqn_nofreeze,
    'pre_graph_11_scratch':
        gconfig.pre_graph_11_scratch,
}


def get_agent_config():
  return DQN, make_networks


def get_tuning_algorithm(input_nodes, existing_ops, search_space,
                         operators, program_length, num_freeze_ops,
                         adjust_loss_weight):
  """Creates the tuning algorithm for pyglove."""
  mutation_probability = FLAGS.mutation_probability

  graph_spec = evolution_v2.GraphSpec(
      template=pg.template(search_space),
      input_nodes=input_nodes,
      existing_ops=existing_ops,
      program_length=program_length,
      operators=operators,
      adjust_loss_weight=adjust_loss_weight,
  )
  graph_hasher = evolution_v2.GraphHasher(input_nodes)
  graph_generator = evolution_v2.GraphGenerator(
      graph_spec, graph_hasher, FLAGS.seed)
  graph_mutator = evolution_v2.GraphMutator(
      graph_spec=graph_spec,
      graph_hasher=graph_hasher,
      mutation_probability=mutation_probability,
      num_freeze_ops=num_freeze_ops)

  return evolution_v2.build_regularized_evolution(
      population_size=FLAGS.population_size,
      tournament_size=FLAGS.tournament_size,
      seed=FLAGS.seed,
      graph_generator=graph_generator,
      graph_mutator=graph_mutator,
  )


def main(_):
  random.seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  tf.random.set_seed(FLAGS.seed)

  env_ids = [
      'CartPole-v0', 'MiniGrid-FourRooms-v0'
  ]
  graph_def = graph_defs[FLAGS.graph_def]
  (operators, input_nodes, existing_ops, search_space, num_freeze_ops,
   program_length) = graph_def()

  generator = get_tuning_algorithm(
      input_nodes,
      existing_ops,
      search_space,
      operators,
      program_length,
      num_freeze_ops,
      adjust_loss_weight=False,
  )

  trial_rewards = []
  for program_spec, feedback in pg.sample(
      search_space, generator, FLAGS.max_trials):  # pytype: disable=wrong-arg-types  # gen-stub-imports

    start_time = time.time()
    agent_cls, network_func = get_agent_config()
    loss_program, valid_program = build_program(
        input_nodes, program_spec, operators, check_path_diff=0)

    graph_viz_link = loss_program.visualize()
    if not valid_program:
      raise InvalidProgramError('Invalid program, infeasible inputs')

    all_metrics = {'final_perf': 0.0}
    agent_cls, network_func = get_agent_config()
    # If duplicate, then don't run training and just return reward.
    if program_spec.duplicate:
      all_metrics['final_perf'] = program_spec.reward
    else:
      for env_idx, env_id in enumerate(env_ids):
        train_results = run_train_loop(
            env_id,
            loss_program,
            agent_cls,
            network_func,
            learning_rate=FLAGS.learning_rate,
            batch_size=FLAGS.batch_size,
            epsilon_schedule=FLAGS.epsilon_schedule,
            use_priority=FLAGS.use_priority,
            limit_steps=False,
            wrapper_version=FLAGS.wrapper_version,
            target_update_period=FLAGS.target_update_period)
        logging.info('train_results: %s', train_results)
        env_perf = train_results[FLAGS.objective_metric]
        all_metrics['final_perf'] += env_perf

        for k, v in train_results.items():
          new_key = '%s_%s' % (env_id, k)
          if new_key not in all_metrics:
            all_metrics[new_key] = float(v)
          else:
            all_metrics[new_key] += v

        # Only run 1st env if performance is bad
        if env_idx == 0 and all_metrics['final_perf'] < 0.6:
          break

    logging.info('all_metrics: %s', all_metrics)
    feedback.add_measurement(
        reward=float(all_metrics['final_perf']),
        metrics=all_metrics,
        step=0,
        elapse_secs=time.time() - start_time,
    )
    feedback.add_link('graph_viz', graph_viz_link)
    feedback.done()
    trial_rewards.append(float(all_metrics['final_perf']))
    if len(trial_rewards) % 10 == 0:
      logging.info('%s, %s', '=' * 80, '\n')
      logging.info('All trial rewards: %s', trial_rewards)


if __name__ == '__main__':
  app.run(main)
