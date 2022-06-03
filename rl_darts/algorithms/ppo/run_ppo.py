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

"""Runs PPO on Procgen."""
import collections
import functools
import os
from absl import app
from absl import flags
from absl import logging

from brain_autorl.rl_darts.algorithms.common import log_util
from brain_autorl.rl_darts.algorithms.common import networks
from brain_autorl.rl_darts.algorithms.ppo import config
from brain_autorl.rl_darts.algorithms.ppo import reverb_utils
from brain_autorl.rl_darts.policies import base_policies
from brain_autorl.rl_darts.procgen import procgen_wrappers

from ml_collections import config_flags
import reverb
import tensorflow as tf

from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.metrics import py_metrics
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import ppo_learner
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import train_utils

flags.DEFINE_string('work_dir', '/tmp/...', 'Work directory for logging.')

FLAGS = flags.FLAGS
config_dict = config.get_config()
config_flags.DEFINE_config_dict(name='config', config=config_dict)


def train_eval(root_dir: str):
  """Train and Evaluates the PPO Agent."""
  test_eval_env = procgen_wrappers.TFAgentsParallelProcGenEnv(
      num_envs=FLAGS.config.num_parallel_actors,
      env_name=FLAGS.config.env_name,
      distribution_mode=FLAGS.config.distribution_mode,
      num_levels=0,
      start_level=0)

  train_eval_env = procgen_wrappers.TFAgentsParallelProcGenEnv(
      num_envs=FLAGS.config.num_parallel_actors,
      env_name=FLAGS.config.env_name,
      distribution_mode=FLAGS.config.distribution_mode,
      num_levels=FLAGS.config.num_levels,
      start_level=0)

  collect_env = procgen_wrappers.TFAgentsParallelProcGenEnv(
      num_envs=FLAGS.config.num_parallel_actors,
      normalize_rewards=True,
      distribution_mode=FLAGS.config.distribution_mode,
      env_name=FLAGS.config.env_name,
      num_levels=FLAGS.config.num_levels,
      start_level=0)

  replay_capacity = FLAGS.config.num_parallel_actors * FLAGS.config.collect_sequence_length

  observation_tensor_spec, action_tensor_spec, time_step_tensor_spec = (
      spec_utils.get_tensor_specs(collect_env))
  observation_tensor_spec = tf.TensorSpec(
      dtype=tf.float32, shape=observation_tensor_spec.shape)

  train_step = train_utils.create_train_step()

  feature_network = base_policies.make_impala_cnn_network(
      depths=FLAGS.config.impala_depths, mlp_size=FLAGS.config.mlp_size)
  if FLAGS.config.use_rnn:
    rnn_cell = tf.keras.layers.LSTMCell(units=FLAGS.config.rnn_hidden_size)
  else:
    rnn_cell = None
  encoder = networks.CustomEncodingNetwork(observation_tensor_spec,
                                           feature_network, rnn_cell)
  actor_net = networks.CustomActorDistributionNetwork(action_tensor_spec,
                                                      encoder)
  value_net = networks.CustomValueNetwork(encoder=encoder)

  agent = ppo_clip_agent.PPOClipAgent(
      time_step_spec=time_step_tensor_spec,
      action_spec=action_tensor_spec,
      optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=FLAGS.config.learning_rate, epsilon=1e-5),
      actor_net=actor_net,
      value_net=value_net,
      # ProcGen uses the original stochastic policy for eval. Empirically, this
      # gives significantly better performance than greedy evaluation.
      greedy_eval=False,
      importance_ratio_clipping=FLAGS.config.importance_ratio_clipping,
      lambda_value=FLAGS.config.lambda_value,
      discount_factor=FLAGS.config.discount_factor,
      entropy_regularization=FLAGS.config.entropy_regularization,
      policy_l2_reg=0.0,
      value_function_l2_reg=0.0,
      shared_vars_l2_reg=0.0,
      value_pred_loss_coef=FLAGS.config.value_pred_loss_coef,
      # This is a legacy argument for the number of times we repeat the data
      # inside of the train function, incompatible with mini batch learning.
      # We set the epoch number from the replay buffer and tf.Data instead.
      num_epochs=FLAGS.config.num_epochs,
      use_gae=FLAGS.config.use_gae,
      use_td_lambda_return=FLAGS.config.use_td_lambda_return,
      normalize_rewards=False,
      reward_norm_clipping=10.0,  # this clips the normalized reward if used
      normalize_observations=FLAGS.config.normalize_observations,
      log_prob_clipping=0.0,
      gradient_clipping=FLAGS.config.gradient_clipping,
      value_clipping=FLAGS.config.value_clipping,
      compute_value_and_advantage_in_train=False,
      update_normalizers_in_train=False,
      debug_summaries=FLAGS.config.debug_summaries,
      summarize_grads_and_vars=FLAGS.config.summarize_grads_and_vars,
      train_step_counter=train_step)
  agent.initialize()

  reverb_server = reverb.Server(
      [
          reverb.Table(  # Replay buffer storing experience for training.
              name='training_table',
              sampler=reverb.selectors.Fifo(),
              remover=reverb.selectors.Fifo(),
              rate_limiter=reverb.rate_limiters.MinSize(1),
              max_size=replay_capacity,
              max_times_sampled=1,
          ),
          reverb.Table(  # Replay buffer storing experience for normalization.
              name='normalization_table',
              sampler=reverb.selectors.Fifo(),
              remover=reverb.selectors.Fifo(),
              rate_limiter=reverb.rate_limiters.MinSize(1),
              max_size=replay_capacity,
              max_times_sampled=1,
          )
      ],
      port=FLAGS.config.reverb_port)

  # Create the replay buffer.
  reverb_replay_train = reverb_replay_buffer.ReverbReplayBuffer(
      agent.collect_data_spec,
      sequence_length=FLAGS.config.collect_sequence_length,
      table_name='training_table',
      server_address='localhost:{}'.format(reverb_server.port),
      # The only collected sequence is used to populate the batches.
      max_cycle_length=1,
      rate_limiter_timeout_ms=100000)
  reverb_replay_normalization = reverb_replay_buffer.ReverbReplayBuffer(
      agent.collect_data_spec,
      sequence_length=FLAGS.config.collect_sequence_length,
      table_name='normalization_table',
      server_address='localhost:{}'.format(reverb_server.port),
      # The only collected sequence is used to populate the batches.
      max_cycle_length=1,
      rate_limiter_timeout_ms=100000)
  rb_observer = reverb_utils.ReverbConcurrentAddBatchObserver(
      reverb_replay_train.py_client, ['training_table', 'normalization_table'],
      sequence_length=FLAGS.config.collect_sequence_length,
      time_step_spec=time_step_tensor_spec,
      stride_length=FLAGS.config.collect_sequence_length,
      allow_multi_episode_sequences=True,
      batch_size=FLAGS.config.num_parallel_actors,
      pad_end_of_episodes=True)

  collect_env_step_metric = py_metrics.EnvironmentSteps()

  def training_dataset_fn():
    return reverb_replay_train.as_dataset(
        sample_batch_size=1,
        num_steps=None,  # keep it None
        sequence_preprocess_fn=agent.preprocess_sequence)

  def normalization_dataset_fn():
    return reverb_replay_normalization.as_dataset(
        sample_batch_size=1,
        num_steps=None,  # keep it None
        sequence_preprocess_fn=agent.preprocess_sequence)

  agent_learner = ppo_learner.PPOLearner(
      root_dir,
      train_step,
      agent,
      experience_dataset_fn=training_dataset_fn,
      normalization_dataset_fn=normalization_dataset_fn,
      num_samples=FLAGS.config.num_parallel_actors,
      num_epochs=FLAGS.config.num_epochs,
      minibatch_size=FLAGS.config.minibatch_size,
      shuffle_buffer_size=FLAGS.config.collect_sequence_length)

  # Re-enable batch_time_steps if environment is not batched.
  collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
      agent.collect_policy,
      use_tf_function=True,
      batch_time_steps=not collect_env.batched)

  collect_actor = actor.Actor(
      collect_env,
      collect_policy,
      train_step,
      steps_per_run=replay_capacity,
      observers=[rb_observer, collect_env_step_metric],
      metrics=actor.collect_metrics(buffer_size=10) + [collect_env_step_metric],
      reference_metrics=[collect_env_step_metric],
      summary_dir=os.path.join(root_dir, learner.TRAIN_DIR),
      summary_interval=FLAGS.config.summary_interval)

  # Re-enable batch_time_steps if environment is not batched.
  eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
      agent.policy,
      use_tf_function=True,
      batch_time_steps=not train_eval_env.batched)

  train_eval_actor = actor.Actor(
      train_eval_env,
      eval_policy,
      train_step,
      metrics=actor.eval_metrics(FLAGS.config.eval_episodes),
      summary_dir=os.path.join(root_dir, 'train_eval'),
      summary_interval=FLAGS.config.summary_interval,
      episodes_per_run=FLAGS.config.eval_episodes)

  test_eval_actor = actor.Actor(
      test_eval_env,
      eval_policy,
      train_step,
      metrics=actor.eval_metrics(FLAGS.config.eval_episodes),
      summary_dir=os.path.join(root_dir, 'test_eval'),
      summary_interval=FLAGS.config.summary_interval,
      episodes_per_run=FLAGS.config.eval_episodes)

  chk = collections.defaultdict(dict)  # Can be checkpointer.
  tf_state = tf.train.Checkpoint(agent=agent, train_step=train_step)
  chk['state']['tf'] = tf_state
  chk['state']['iteration'] = 0

  metric_logger = log_util.MetricLogger(root_dir)
  logging.info('Training.')
  while chk['state']['iteration'] < FLAGS.config.num_iterations:
    collect_actor.run()
    metric_logger.log_metrics(collect_actor.metrics, chk['state']['iteration'],
                              'collect_actor')

    rb_observer.reset()
    agent_learner.run()
    reverb_replay_train.clear()
    reverb_replay_normalization.clear()

    if chk['state']['iteration'] % FLAGS.config.eval_interval == 0:
      logging.info('Evaluating.')
      train_eval_actor.run_and_log()
      metric_logger.log_metrics(train_eval_actor.metrics,
                                chk['state']['iteration'], 'train_eval_actor')
      test_eval_actor.run_and_log()
      metric_logger.log_metrics(test_eval_actor.metrics,
                                chk['state']['iteration'], 'test_eval_actor')

    chk['state']['iteration'] += 1

  rb_observer.close()
  reverb_server.stop()


def main(_):
  logging.set_verbosity(logging.INFO)
  tf.compat.v1.enable_v2_behavior()
  tf.gfile.MakeDirs(FLAGS.work_dir)

  train_eval(root_dir=FLAGS.work_dir)


if __name__ == '__main__':
  multiprocessing.handle_main(functools.partial(app.run, main))
