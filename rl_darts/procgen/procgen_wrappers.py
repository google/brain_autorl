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

# pytype: disable=wrong-arg-types
"""Wrappers for ProcGen, modified for current Acme + TFAgents setup."""
import datetime
from multiprocessing import dummy as mp_threads
from typing import Sequence

from acme.wrappers import gym_wrapper as acme_gym_wrapper
from baselines.common import vec_env

from brain_autorl.rl_darts.procgen import common_wrappers

from dm_env import specs
import gym
import numpy as np
from procgen import env as procgen_env
from procgen import scalarize
import tensorflow as tf

from tf_agents import environments as tf_agents_environments
from tf_agents.environments import gym_wrapper as tf_agents_gym_wrapper
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import nest_utils

ENV_NAMES = procgen_env.ENV_NAMES


class ProcGenRewardNormalizer(gym.RewardWrapper):
  """Normalizes the rewards by dividing by the maximum achieveable reward.

  This is needed because algorithms such as DQN require reward clipping.
  Ideally, doing this should not obfuscate logged results, as the maximum
  episodic would now be 1.0. The constants are found in Appendix C of
  https://arxiv.org/pdf/1912.01588.pdf.
  """

  def __init__(self, env, env_name):
    self.env = env
    self.env_name = env_name
    self.action_space = self.env.action_space
    self.observation_space = self.env.observation_space
    self.metadata = self.env.metadata
    self.maximum_rewards_obvious = {
        "coinrun": 10.0,
        "dodgeball": 19.0,
        "miner": 20.0,
        "jumper": 10.0,
        "leaper": 10.0,
        "maze": 10.0,
        "bigfish": 40.0,
        "heist": 10.0,
        "plunder": 30.0,
        "ninja": 10.0,
        "bossfight": 13.0
    }

    self.maximum_rewards_empirical = {
        "caveflyer": 13.4,
        "chaser": 14.2,
        "climber": 12.6,
        "starpilot": 35.0,
        "fruitbot": 27.2
    }

  def reward(self, reward):
    if self.env_name in self.maximum_rewards_obvious:
      return reward / self.maximum_rewards_obvious[self.env_name]
    elif self.env_name in self.maximum_rewards_empirical:
      return reward / self.maximum_rewards_empirical[self.env_name]
    else:
      raise ValueError("Environment Name not in supported dictionaries.")


def vector_wrap_environment(venv,
                            normalize_obs=False,
                            normalize_ret=False,
                            monitor=True):
  """Converts observation dicts with key 'rgb' into just array.

  Can also log files and normalize observations.

  Args:
    venv: Original venv.
    normalize_obs: Normalizes the observation using (x-mean)/std.
    normalize_ret: Normalizes the reward using x/std. Should generally NOT be
      used. Use ProcGenRewardNormalizer instead.
    monitor: Monitors metadata (usually only used inside OpenAI PPO).

  Returns:
    Wrapped venv.

  """
  venv = vec_env.VecExtractDictObs(venv, "rgb")

  if monitor:
    venv = vec_env.VecMonitor(
        venv=venv,
        filename=None,
        keep_buf=100,
    )
  if normalize_obs or normalize_ret:
    venv = vec_env.VecNormalize(
        venv=venv, ob=normalize_obs, ret=normalize_ret, clipob=10., cliprew=10.)
  return venv


class AcmeProcgenEnv(acme_gym_wrapper.GymWrapper):
  """dm_env wrapper for ProcGen (scalar), with additional vectorized environment wrapping.

  Note that the output of ProcgenEnv() is a vectorized environment (i.e. there's
  a batch dimension equal to num_envs, where a batched set of actions are
  expected), which is normally used for
  multithreading. We disable this via setting
  num_envs=1, and applying Scalarize().
  """

  def __init__(self, env_name, seed=0, distribution_mode="hard", **kwargs):
    self.current_seed = seed

    def make_env(rand_seed):
      output_env = procgen_env.ProcgenEnv(
          num_envs=1,
          num_threads=0,
          env_name=env_name,
          rand_seed=rand_seed,
          distribution_mode=distribution_mode,
          **kwargs)
      output_env = vector_wrap_environment(
          output_env, normalize_obs=False, normalize_ret=False, monitor=False)
      output_env = ProcGenRewardNormalizer(output_env, env_name=env_name)
      output_env = common_wrappers.ObsToFloat(output_env, divisor=255.0)
      output_env = scalarize.Scalarize(output_env)
      return output_env

    self._make_env = make_env
    super().__init__(make_env(self.current_seed))  # pytype: disable=wrong-arg-types

  def reset(self):
    """Implements dm_env.Environment.reset."""
    self.current_seed = int(datetime.datetime.utcnow().timestamp()) % 100000
    if not self._reset_next_step:
      self.gym_env.close()
      self.gym_env = self._make_env(self.current_seed)
    return super().reset()

  def reward_spec(self):
    """Implements dm_env.Environment.reward_spec."""
    return specs.Array(shape=(), dtype=np.float32, name="reward")


class TFAgentsParallelProcGenEnv(tf_agents_environments.PyEnvironment):
  """Wrapped ProcGen environment for TF_agents algorithms."""

  def __init__(self,
               num_envs,
               discount=1.0,
               spec_dtype_map=None,
               simplify_box_bounds=True,
               flatten=False,
               normalize_rewards=False,
               **procgen_kwargs):
    """Uses Native C++ Environment Vectorization, which reduces RAM usage.

    Except the num_envs and **procgen_kwargs, all of the other __init__
    args come from the original TF-Agents GymWrapper and
    ParallelPyEnvironment wrappers.

    Args:
      num_envs: List of callables that create environments.
      discount: Discount rewards automatically (also done in algorithms).
      spec_dtype_map: A dict from spaces to dtypes to use as the default dtype.
      simplify_box_bounds: Whether to replace bounds of Box space that are
        arrays with identical values with one number and rely on broadcasting.
      flatten: Boolean, whether to use flatten action and time_steps during
        communication to reduce overhead.
      normalize_rewards: Use VecNormalize to normalize rewards. Should be used
        for collect env only.
      **procgen_kwargs: Keyword arguments passed into the native ProcGen env.
    """
    super(TFAgentsParallelProcGenEnv, self).__init__()

    self._num_envs = num_envs

    parallel_env = procgen_env.ProcgenEnv(num_envs=num_envs, **procgen_kwargs)
    parallel_env = vector_wrap_environment(
        parallel_env,
        normalize_obs=False,
        normalize_ret=normalize_rewards,
        monitor=False)
    parallel_env = common_wrappers.ObsToFloat(parallel_env, divisor=255.0)

    self._parallel_env = parallel_env

    self._observation_spec = tf_agents_gym_wrapper.spec_from_gym_space(
        self._parallel_env.observation_space, spec_dtype_map,
        simplify_box_bounds, "observation")

    self._action_spec = tf_agents_gym_wrapper.spec_from_gym_space(
        self._parallel_env.action_space, spec_dtype_map, simplify_box_bounds,
        "action")
    self._time_step_spec = ts.time_step_spec(self._observation_spec,
                                             self.reward_spec())

    self._flatten = flatten
    self._discount = discount

    self._dones = [True] * num_envs  # Contains "done"s for all subenvs.

  @property
  def parallel_env(self):
    return self._parallel_env

  @property
  def batched(self) -> bool:
    return True

  @property
  def batch_size(self) -> int:
    return self._num_envs

  def observation_spec(self) -> types.NestedArraySpec:
    return self._observation_spec

  def action_spec(self) -> types.NestedArraySpec:
    return self._action_spec

  def time_step_spec(self) -> ts.TimeStep:
    return self._time_step_spec

  def close(self):
    self._parallel_env.close()

  def _step(self, actions):
    if tf.is_tensor(actions):
      actions = actions.numpy()
    observations, rewards, temp_dones, self._infos = self._parallel_env.step(
        actions)
    timesteps = []

    for i, done in enumerate(temp_dones):
      if done:
        time_step = ts.termination(observations[i], rewards[i])
      else:
        if self._dones[i]:
          time_step = ts.restart(observations[i])
        else:
          time_step = ts.transition(observations[i], rewards[i], self._discount)
      timesteps.append(time_step)

    self._dones = temp_dones

    return self._stack_time_steps(timesteps)

  def _reset(self):
    observations = self._parallel_env.reset()
    self._dones = [False] * self._num_envs

    timesteps = ts.restart(observations, batch_size=self._num_envs)
    return timesteps

  def _stack_time_steps(self, time_steps):
    """Given a list of TimeStep, combine to one with a batch dimension."""
    if self._flatten:
      return nest_utils.fast_map_structure_flatten(
          lambda *arrays: np.stack(arrays), self._time_step_spec, *time_steps)
    else:
      return nest_utils.fast_map_structure(lambda *arrays: np.stack(arrays),
                                           *time_steps)


class CustomMultiGameProcGenEnv(tf_agents_environments.PyEnvironment):
  """Custom implementation that concatenates multiple (already parallelized + batched) Procgen games into one batched environment.

  Useful for joint training on multiple games. Uses thread-level parallelism
  currently.
  """

  def __init__(self,
               env_names: Sequence[str],
               num_envs_per_game: int,
               multithreading: bool = True,
               **procgen_kwargs):
    self._env_names = env_names
    self._num_envs_per_game = num_envs_per_game
    self._multithreading = multithreading
    self._envs = [
        TFAgentsParallelProcGenEnv(
            num_envs=num_envs_per_game, env_name=env_name, **procgen_kwargs)
        for env_name in env_names
    ]
    if self._multithreading:
      self._pool = mp_threads.Pool(self.num_games)

  def _execute(self, fn, iterable):
    if self._multithreading:
      return self._pool.map(fn, iterable)
    else:
      return [fn(x) for x in iterable]

  @property
  def batched(self) -> bool:
    return True

  @property
  def batch_size(self) -> int:
    return self.num_games * self._num_envs_per_game

  @property
  def num_games(self) -> int:
    return len(self._env_names)

  def _step(self, actions: types.NestedArray) -> ts.TimeStep:
    split_game_actions = self._split_batched_actions(actions)
    split_game_time_steps = self._execute(
        lambda game_and_actions: game_and_actions[0].step(game_and_actions[1]),
        zip(self._envs, split_game_actions))

    return self._concat_list_batched_time_steps(split_game_time_steps)

  def _reset(self) -> ts.TimeStep:
    split_game_time_steps = []
    for env in self._envs:
      split_game_time_steps.append(env.reset())
    return self._concat_list_batched_time_steps(split_game_time_steps)

  def observation_spec(self) -> types.NestedArraySpec:
    return self._envs[0].observation_spec()

  def action_spec(self) -> types.NestedArraySpec:
    return self._envs[0].action_spec()

  def _split_batched_actions(
      self, batched_actions: types.NestedArray) -> types.NestedArray:
    """Splits actions into the corresponding (already batched) game actions."""
    split_game_actions = []
    for i in range(self.num_games):
      specific_game_batched_actions = batched_actions[i *
                                                      self._num_envs_per_game:
                                                      (i + 1) *
                                                      self._num_envs_per_game]
      split_game_actions.append(specific_game_batched_actions)
    return split_game_actions

  def _concat_list_batched_time_steps(
      self,
      batched_time_steps: Sequence[types.NestedArray]) -> types.NestedArray:
    unstacked_single_time_steps = []
    for game_time_steps in batched_time_steps:
      unstacked_single_time_steps.extend(
          nest_utils.unstack_nested_arrays(game_time_steps))
    return nest_utils.stack_nested_arrays(unstacked_single_time_steps)


class TFAgentsSingleProcessProcGenEnv(tf_agents_gym_wrapper.GymWrapper):
  """Single Process Environment (legacy) for TFAgents. Useful for debugging."""

  def __init__(self, env_name, **kwargs):

    output_env = procgen_env.ProcgenEnv(
        num_envs=1, num_threads=0, env_name=env_name, **kwargs)
    output_env = vector_wrap_environment(
        output_env, normalize_obs=False, normalize_ret=False, monitor=False)
    output_env = common_wrappers.ObsToFloat(output_env, divisor=255.0)
    output_env = scalarize.Scalarize(output_env)

    super(TFAgentsSingleProcessProcGenEnv, self).__init__(gym_env=output_env)
