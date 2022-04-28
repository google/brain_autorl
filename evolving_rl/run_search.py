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
import time

from absl import app
from absl import flags
from brain_autorl.evolving_rl.custom_dqn import DQN
from brain_autorl.evolving_rl.custom_dqn import make_networks
from brain_autorl.evolving_rl.env_utils import run_train_loop
from brain_autorl.evolving_rl.evolution import CGSRegularizedEvolution
from brain_autorl.evolving_rl.evolution import DAGPointMutator
import brain_autorl.evolving_rl.graph_configs as gconfig
from brain_autorl.evolving_rl.program import build_program
from brain_autorl.evolving_rl.program import InvalidProgramError
import pyglove.google as pg


FLAGS = flags.FLAGS
flags.DEFINE_integer('max_trials', 100, 'Max number of vizier trials.')
flags.DEFINE_string('objective_metric', 'train/normalized_avg_return_last50',
                    'Objective to maximize')
flags.DEFINE_enum('tuning_algo', 'evolution', ['default', 'ppo', 'evolution'],
                  'Tuning algorithm to use.')
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


def get_tuning_algorithm(tuning_algo, input_nodes, existing_ops, search_space,
                         operators, program_length, num_freeze_ops,
                         adjust_loss_weight):
  """Creates the tuning algorithm for pyglove."""
  if tuning_algo == 'ppo':
    return pg.policy_gradient.PPO(
        update_batch_size=8, num_updates_per_feedback=10)
  elif tuning_algo == 'evolution':
    mutation_probability = 0.95
    template = pg.template(search_space)
    mutator = DAGPointMutator(
        mutation_probability=mutation_probability,
        input_nodes=input_nodes,
        num_freeze_ops=num_freeze_ops)
    mutator.template = template
    mutator.operators = operators
    mutator.existing_ops = existing_ops
    mutator.program_length = program_length
    mutator.adjust_loss_weight = adjust_loss_weight
    algorithm = CGSRegularizedEvolution(
        population_size=FLAGS.population_size,
        tournament_size=FLAGS.tournament_size,
        mutator=mutator)
    return algorithm
  else:
    raise ValueError(f'tuning algorithm {tuning_algo} not supported')


def main(_):
  env_ids = [
      'CartPole-v0', 'MiniGrid-KeyCorridorS3R1-v0',
      'MiniGrid-Dynamic-Obstacles-6x6-v0', 'MiniGrid-DoorKey-5x5-v0',
      'MiniGrid-MultiRoom-N2-S4-v0', 'MiniGrid-FourRooms-v0'
  ]
  graph_def = graph_defs[FLAGS.graph_def]
  (operators, input_nodes, existing_ops, search_space, num_freeze_ops,
   program_length) = graph_def()

  generator = get_tuning_algorithm(
      'evolution',
      input_nodes,
      existing_ops,
      search_space,
      operators,
      program_length,
      num_freeze_ops,
      adjust_loss_weight=False,
  )

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
        print('train_results', train_results)
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

    print('all_metrics', all_metrics)
    feedback.add_measurement(
        reward=float(all_metrics['final_perf']),
        metrics=all_metrics,
        step=0,
        elapse_secs=time.time() - start_time,
    )
    feedback.add_link('graph_viz', graph_viz_link)


if __name__ == '__main__':
  app.run(main)
