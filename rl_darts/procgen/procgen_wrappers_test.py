"""Tests for procgen.procgen_wrappers."""
# pylint: disable=g-long-lambda
from absl import logging
from absl.testing import parameterized

from brain_autorl.rl_darts.procgen import procgen_wrappers

import numpy as np
import tensorflow as tf

from tf_agents.environments import parallel_py_environment
from tf_agents.system import system_multiprocessing as multiprocessing


class ProcgenWrappersTest(tf.test.TestCase, parameterized.TestCase):

  def test_acme_procgen_wrapper(self):
    env = procgen_wrappers.AcmeProcgenEnv(
        env_name='coinrun',
        seed=0,
        distribution_mode='easy',
        num_levels=200,
        start_level=0)

    timestep = env.reset()
    obs = timestep.observation
    logging.info('Obs: %s', obs)
    logging.info('Obs Shape: %s', obs.shape)

  @parameterized.parameters({'env_names': ['coinrun']},
                            {'env_names': ['coinrun', 'starpilot']})
  def test_multi_game_procgen_env(self, env_names):
    num_envs_per_game = 2
    correct_batch_size = len(env_names) * num_envs_per_game
    num_steps = 1200

    multi_game_procgen_env = procgen_wrappers.CustomMultiGameProcGenEnv(
        env_names=env_names, num_envs_per_game=num_envs_per_game)

    ts = multi_game_procgen_env.reset()
    self.assertEqual(ts.observation.shape, (correct_batch_size, 64, 64, 3))

    for _ in range(num_steps):
      action = np.random.randint(
          low=0, high=15, size=(len(env_names) * num_envs_per_game))
      ts = multi_game_procgen_env.step(action)
      self.assertEqual(ts.observation.shape, (correct_batch_size, 64, 64, 3))

  def test_tf_agents_equivalence(self):
    """Tests for equivalence between ParallelPyEnv and Native Batched Env."""
    env_name = 'coinrun'
    rand_seed = 0
    start_level = 0
    distribution_mode = 'easy'
    num_envs = 1
    num_steps = 1200

    env1 = procgen_wrappers.TFAgentsParallelProcGenEnv(
        num_envs=num_envs,
        env_name=env_name,
        start_level=start_level,
        rand_seed=rand_seed,
        distribution_mode=distribution_mode)

    lambdas = [
        lambda: procgen_wrappers.TFAgentsSingleProcessProcGenEnv(
            env_name=env_name,
            rand_seed=rand_seed,
            start_level=start_level,
            distribution_mode=distribution_mode)
    ] * num_envs
    env2 = parallel_py_environment.ParallelPyEnvironment(lambdas)

    ts1 = env1.reset()
    ts2 = env2.reset()

    self.assertAllEqual(ts1.observation, ts2.observation)
    self.assertEqual(ts1.step_type, ts2.step_type)

    for i in range(num_steps):
      logging.info('i: %d', i)
      action = np.random.randint(low=0, high=15, size=(1))
      ts1 = env1.step(action)
      ts2 = env2.step(action)

      logging.info('TS1: %s', ts1.step_type)
      logging.info('TS2: %s', ts2.step_type)
      self.assertEqual(ts1.step_type, ts2.step_type)

      logging.info('TS1 reward: %s', ts1.reward)
      logging.info('TS2 reward: %s', ts2.reward)
      self.assertEqual(ts1.reward, ts2.reward)

      # Disabled due to strange randomness in background images.
      # self.assertAllEqual(ts1.observation, ts2.observation)


if __name__ == '__main__':
  multiprocessing.handle_test_main(tf.test.main)
