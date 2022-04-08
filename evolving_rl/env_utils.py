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

"""Environment training loops for running the RL loss function evaluation.

run_train_loop will take an environment and loss function and evaluate the
loss function on that environment by training an RL agent.
"""
# pytype: skip-file
import functools
import itertools
import math
import operator
import time
import traceback
from typing import Optional

from absl import flags
import acme
from acme import specs
from acme import wrappers
from acme.tf import utils as tf2_utils
from acme.utils import loggers
from brain_autorl.evolving_rl.env_config import atari_env_config
from brain_autorl.evolving_rl.env_config import full_env_config
from bsuite import bsuite
import gym
from gym_minigrid.minigrid import COLOR_TO_IDX
from gym_minigrid.minigrid import OBJECT_TO_IDX
from gym_minigrid.minigrid import STATE_TO_IDX
import numpy as np
from PIL import Image
import tensorflow as tf


def compute_metrics(env, env_id, eps_returns, label):
  """Compute the performance metrics given the episode returns for an env."""
  max_return = full_env_config[env_id]['max_return']
  min_return = full_env_config[env_id]['min_return']
  spread = max_return - min_return
  metrics = {}
  eps_returns_np = np.array(eps_returns)
  metrics['final_return'] = eps_returns[-1]
  metrics['final_return_last10'] = sum(eps_returns[-10:]) / 10.
  metrics['avg_return'] = eps_returns_np.mean()
  metrics['std_return'] = eps_returns_np.std()
  metrics['normalized_avg_return'] = (
      (np.clip(eps_returns_np, min_return, max_return) - min_return) /
      spread).mean()
  metrics['normalized_avg_return_last10'] = (
      (np.clip(eps_returns_np[-10:], min_return, max_return) - min_return) /
      spread).mean()
  metrics['normalized_avg_return_last50'] = (
      (np.clip(eps_returns_np[-50:], min_return, max_return) - min_return) /
      spread).mean()
  if check_env_id_bsuite(env_id):
    score = env.bsuite_info()[full_env_config[env_id]['info_key']]
    score = score / full_env_config[env_id]['num_episodes']
    # Convert avg regret to avg return for umbrella where regret of 2 is worst
    if 'umbrella' in env_id:
      metrics['normalized_avg_return'] = 1 - (score / 2.)
    else:
      score = (score - min_return) / spread
      metrics['normalized_avg_return'] = score

    metrics['normalized_avg_return_last50'] = metrics['normalized_avg_return']

  label_metrics = {}
  for k, v in metrics.items():
    label_metrics['%s/%s' % (label, k)] = v
  return label_metrics


def run_train_loop(env_id,
                   loss_program,
                   agent_cls,
                   network_func,
                   learning_rate=1e-4,
                   batch_size=32,
                   epsilon_schedule=100000,
                   wrapper_version='v3',
                   target_update_period=100,
                   limit_steps=False,
                   mlp_size=256,
                   samples_per_insert=32,
                   use_priority=False,
                   epsilon_final_p=0.02,
                   min_replay_size=1000,
                   record_qval=False,
                   log_to_xm=True,
                   force_reproducibility=False,
                   generate_eval_metrics=False,):
  """Evaluates a loss program on an environment."""
  env_seed = 1 if force_reproducibility else None
  env = make_environment_gym(
      env_id, minigrid_obs_version=wrapper_version, seed=env_seed)
  env_spec = specs.make_environment_spec(env)
  eval_env = make_environment_gym(
      env_id,
      minigrid_obs_version=wrapper_version,
      evaluation=True,
      seed=env_seed)

  networks = network_func(env_id, env_spec, mlp_size)
  env_config = full_env_config[env_id]
  if 'epsilon_schedule' in env_config:
    epsilon_schedule = env_config['epsilon_schedule']
  # if 'target_update_period' in env_config:
  #  target_update_period = env_config['target_update_period']
  if 'samples_per_insert' in env_config:
    samples_per_insert = env_config['samples_per_insert']
  if 'learning_rate' in env_config:
    learning_rate = env_config['learning_rate']
  if 'epsilon_final_p' in env_config:
    epsilon_final_p = env_config['epsilon_final_p']
  if 'min_replay_size' in env_config:
    min_replay_size = env_config['min_replay_size']
  noop = env_config.get('noop', False)
  checkpoint = env_config.get('checkpoint', False)
  if 'eval_period' in env_config:
    eval_period = env_config['eval_period']
  else:
    eval_period = None

  print('Num Actions', env_spec.actions.num_values)
  print(env_spec.actions)
  try:
    # Construct the agent.
    kwargs = networks
    if force_reproducibility:
      # Assume we are using the custom DQN.
      # We don't direct control of reverb's sampling (e.g. the RNG). Using the
      # in-memory replay ensures a consistent RNG is used. The actor_sample_seed
      # controls the seed used for sampling from epsilon-greedy distribution.
      kwargs.update({'use_reverb': False, 'actor_sample_seed': 0})
    train_agent = agent_cls(
        environment_spec=env_spec,
        learning_rate=learning_rate,
        checkpoint=checkpoint,
        n_step=1,
        batch_size=batch_size,
        loss_program=loss_program,
        epsilon_initial_p=1.0,
        epsilon_final_p=epsilon_final_p,
        epsilon_schedule=epsilon_schedule,
        target_update_period=target_update_period,
        samples_per_insert=samples_per_insert,
        use_priority=use_priority,
        min_replay_size=min_replay_size,
        **kwargs)
    # If logger is left None, underlying library creates an xm logger.
    train_logger = None if log_to_xm else loggers.NoOpLogger()
    train_loop = EnvironmentTrainLoopWithResults(
        env, train_agent, label='train_loop', logger=train_logger)
    train_returns = []

    eval_agent = agent_cls(
        environment_spec=env_spec,
        n_step=1,
        loss_program=loss_program,
        checkpoint=False,
        epsilon_initial_p=0.001,
        epsilon_final_p=0.001,
        max_replay_size=1000,
        **networks)
    eval_logger = None if log_to_xm else loggers.NoOpLogger()
    eval_loop = EnvironmentEvalLoopWithResults(
        eval_env, eval_agent, label='eval_loop', logger=eval_logger)
    eval_returns = []
    num_steps = 0

    def run_train(train_loop):
      ep_result = train_loop.run(
          num_episodes=1, record_qval=record_qval, noop=noop)
      train_returns.append(ep_result['episode_return'])
      return ep_result['episode_length']

    def run_eval(eval_loop, num_steps):
      ep_result = eval_loop.run(
          int(env_config['eval_episodes']), num_steps, noop)
      eval_returns.append(ep_result['episode_return'])

    num_steps = 0
    prev_steps = 0
    if (limit_steps and
        'num_steps' in env_config) or 'train_steps' in env_config:
      max_steps = env_config['num_steps'] if limit_steps else env_config[
          'train_steps']
      while num_steps < max_steps:
        train_loop_steps = run_train(train_loop)
        # Check if passed interval of eval_period
        if generate_eval_metrics and eval_period and num_steps % eval_period > (
            num_steps + train_loop_steps) % eval_period:
          run_eval(eval_loop, num_steps - prev_steps)
          prev_steps = num_steps
        num_steps += train_loop_steps

    else:
      num_episodes = int(env_config['num_episodes'])
      for _ in range(num_episodes):
        num_steps += run_train(train_loop)

    train_metrics = compute_metrics(env, env_id, train_returns, 'train')

    if generate_eval_metrics:
      run_eval(eval_loop, num_steps)
      eval_metrics = compute_metrics(eval_env, env_id, eval_returns, 'eval')
    else:
      eval_metrics = {}

    del train_loop
    del eval_loop
    del train_agent
    del eval_agent
  except AssertionError:
    print(traceback.format_exc())
    eval_metrics = {
        'eval/normalized_avg_return': 0.0,
        'train/normalized_avg_return': 0.0,
        'train/normalized_avg_return_last10': 0.0,
        'train/normalized_avg_return_last50': 0.0
    }
    train_metrics = {}
  return {**train_metrics, **eval_metrics}


class LinearSchedule(object):
  """Linear schedule for epsilon greedy values."""

  def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
    """Initialize linear schedule.

    Linear interpolation between initial_p and final_p over
    schedule_timesteps. After this many timesteps pass final_p is
    returned.
    Args:
      schedule_timesteps: Number of timesteps for which to linearly anneal
        initial_p to final_p.
      final_p: final output value.
      initial_p: initial output value.
    """
    self.schedule_timesteps = schedule_timesteps
    self.final_p = final_p
    self.initial_p = initial_p

  def value(self, t):
    """Computes the epsilon value at a given time."""
    fraction = min(float(t) / self.schedule_timesteps, 1.0)
    return self.initial_p + fraction * (self.final_p - self.initial_p)


class EnvironmentTrainLoopWithResults(acme.EnvironmentLoop):
  """A simple RL environment loop modifed from Acme to return results."""

  def run(self,
          num_episodes: Optional[int] = None,
          record_qval=False,
          noop=False):
    iterator = range(num_episodes) if num_episodes else itertools.count()

    qvals = []
    extra_info = {}
    for _ in iterator:
      # Reset any counts and start the environment.
      start_time = time.time()
      episode_steps = 0
      episode_return = 0
      timestep = self._environment.reset()

      # Make the first observation.
      self._actor.observe_first(timestep)

      if record_qval:
        _, qvals = self._actor.select_action(timestep.observation)

      noop_idx = np.random.randint(0, 30)
      # Run an episode.
      while not timestep.last():
        # Generate an action from the agent's policy and step the environment.
        action, _ = self._actor.select_action(timestep.observation)
        if noop and episode_steps < noop_idx:
          action = np.array([0]).astype(np.int32)[0]
        timestep = self._environment.step(action)
        # Have the agent observe the timestep and let the actor update itself.

        self._actor.observe(action, next_timestep=timestep)
        self._actor.update()

        # Book-keeping.
        episode_steps += 1
        episode_return += timestep.reward

      # Record counts.
      counts = self._counter.increment(episodes=1, steps=episode_steps)

      if record_qval and self._counter.get_counts()['steps'] > 1000:
        extra_info = self._actor._learner.get_info()  # pylint: disable=protected-access

      # Collect the results and combine with counts.
      steps_per_second = episode_steps / (time.time() - start_time)
      result = {
          'episode_length': episode_steps,
          'episode_return': episode_return,
          'steps_per_second': steps_per_second,
      }
      for i, qval in enumerate(qvals):
        result['qval_%d' % i] = qval
      for k, v in extra_info.items():
        result[k] = tf2_utils.to_numpy_squeeze(v)

      result.update(counts)

      # Log the given results.
      self._logger.write(result)
    return result


class EnvironmentEvalLoopWithResults(acme.EnvironmentLoop):
  """A simple RL environment loop modifed from Acme to return results."""

  def run(self, num_episodes: Optional[int] = None, num_steps=None, noop=False):
    iterator = range(num_episodes) if num_episodes else itertools.count()

    episode_returns = []
    episode_steps_lst = []
    start_time = time.time()
    for _ in iterator:
      # Reset any counts and start the environment.
      episode_steps = 0
      episode_return = 0
      timestep = self._environment.reset()

      # Make the first observation.
      self._actor.observe_first(timestep)
      noop_idx = np.random.randint(0, 30)
      # Run an episode.
      while not timestep.last():
        # Generate an action from the agent's policy and step the environment.
        action, _ = self._actor.select_action(timestep.observation)
        if noop and episode_steps < noop_idx:
          action = np.array([0]).astype(np.int32)[0]
        timestep = self._environment.step(action)
        # Have the agent observe the timestep and let the actor update itself.
        self._actor.observe(action, next_timestep=timestep)

        # Book-keeping.
        episode_steps += 1
        episode_return += timestep.reward

      episode_returns.append(episode_return)
      episode_steps_lst.append(episode_steps)

    # Record counts.
    counts = self._counter.increment(episodes=num_episodes, steps=num_steps)

    # Collect the results and combine with counts.
    steps_per_second = sum(episode_steps_lst) / (time.time() - start_time)
    result = {
        'episode_length': int(np.mean(episode_steps_lst)),
        'episode_return': float(np.mean(episode_returns)),
        'steps_per_second': steps_per_second,
    }
    result.update(counts)

    # Log the given results.
    self._logger.write(result)
    return result


class FloatRewardWrapper(gym.core.RewardWrapper):
  """Wraps the reward to float, which is standard for `dm_env.Environment`."""

  def reward(self, reward):
    return float(reward)


class ScaleR10Wrapper(gym.core.RewardWrapper):
  """Wraps the reward to float, which is standard for `dm_env.Environment`."""

  def reward(self, reward):
    return float(reward) * 10.


class ScaleR100Wrapper(gym.core.RewardWrapper):
  """Wraps the reward to float, which is standard for `dm_env.Environment`."""

  def reward(self, reward):
    return float(reward) * 100.


class ScaleR1000Wrapper(gym.core.RewardWrapper):
  """Wraps the reward to float, which is standard for `dm_env.Environment`."""

  def reward(self, reward):
    return float(reward) * 1000.


class ScaleR10000Wrapper(gym.core.RewardWrapper):
  """Wraps the reward to float, which is standard for `dm_env.Environment`."""

  def reward(self, reward):
    return float(reward) * 10000.


class FlatObsWrapper(gym.core.ObservationWrapper):
  """Wraps observed image and the mission string into one flat array.

    Encode mission strings using a one-hot scheme, and combine these with
    observed images into one flat array.

    This is the same as `FlatObsWrapper` in
    third_party/py/gym_minigrid/wrappers.py, except two changes:
    1) Observation dtype is float32 instead of uint8, and the values are
    normalized to [0, 1] instead of [0, 255].
    2) The returned `obs` is reshaped to match the shape defined
    in `self.observation_space`. This is to make sure when we add an observation
    into reverb, it matches the signature reverb expects (which is inferred from
    the `observation_space` here.
  """

  def __init__(self, env):
    super().__init__(env)
    new_img_size = (32, 32, 3)
    img_size = functools.reduce(operator.mul, new_img_size, 1)

    self.observation_space = gym.spaces.Box(
        low=0,
        high=1.0,
        shape=(img_size,),  # number of cells
        dtype='float32')

  def observation(self, obs):
    image = obs['image'] / 255.
    image = tf.image.resize(image, (32, 32))
    return image.numpy().flatten()


class FlatObsWrapperV2(gym.core.ObservationWrapper):
  """Wraps observed image and the mission string into one flat array."""

  def __init__(self, env):
    super().__init__(env)
    new_img_size = (12, 12, 3)
    img_size = functools.reduce(operator.mul, new_img_size, 1)

    self.observation_space = gym.spaces.Box(
        low=0,
        high=1.0,
        shape=(img_size,),  # number of cells
        dtype='float32')

  def observation(self, obs):
    image = obs['image'] / 255.
    image = tf.image.resize(image, (12, 12))
    return image.numpy().flatten()


class OneHotPartialObsWrapper(gym.core.ObservationWrapper):
  """Wrapper for minigrid partially observed onehot.

  Wrapper to get a one-hot encoding of a partially observable
  agent view as observation.
  """

  def __init__(self, env, tile_size=8):
    super().__init__(env)

    self.tile_size = tile_size

    obs_shape = env.observation_space['image'].shape

    # Number of bits per cell
    num_bits = len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + len(STATE_TO_IDX)
    self._observation_space = gym.spaces.Box(
        low=0,
        high=255,
        shape=(
            obs_shape[0],
            obs_shape[1],
            num_bits,
        ),
        dtype='float32')
    self.observation_space = gym.spaces.Box(
        low=0,
        high=255,
        shape=(obs_shape[0] * obs_shape[1] * num_bits,),
        dtype='float32')

  def observation(self, obs):
    img = obs['image']
    out = np.zeros(self._observation_space.shape, dtype='uint8')

    for i in range(img.shape[0]):
      for j in range(img.shape[1]):
        o_type = img[i, j, 0]
        color = img[i, j, 1]
        state = img[i, j, 2]

        out[i, j, o_type] = 1
        out[i, j, len(OBJECT_TO_IDX) + color] = 1
        out[i, j, len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + state] = 1
    return out.flatten().astype(np.float32)


class FullyObsWrapper(gym.core.ObservationWrapper):
  """Fully observable gridworld using a compact grid encoding."""

  def __init__(self, env):
    super().__init__(env)

    self.observation_space = gym.spaces.Box(
        low=0,
        high=255,
        shape=(self.env.width * self.env.height * 3,),  # number of cells
        dtype='float32')

  def observation(self, obs):
    env = self.unwrapped
    full_grid = env.grid.encode()
    full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array(
        [OBJECT_TO_IDX['agent'], COLOR_TO_IDX['red'], env.agent_dir])

    return full_grid.astype(np.float32).flatten()


class FullyObsWrapperV2(gym.core.ObservationWrapper):
  """Fully observable gridworld using a compact grid encoding.

  Removes walls for efficiency.
  """

  def __init__(self, env):
    super().__init__(env)

    self.observation_space = gym.spaces.Box(
        low=0,
        high=255,
        shape=((self.env.width - 2) * (self.env.height - 2) *
               3,),  # number of cells
        dtype='float32')

  def observation(self, obs):
    env = self.unwrapped
    full_grid = env.grid.encode()
    full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array(
        [OBJECT_TO_IDX['agent'], COLOR_TO_IDX['red'], env.agent_dir])
    full_grid = full_grid[1:-1, 1:-1]
    return full_grid.astype(np.float32).flatten()


class FullyObsWrapperV3(gym.core.ObservationWrapper):
  """Fully observable gridworld using a compact grid encoding.

  Normalizes output.
  """

  def __init__(self, env):
    super().__init__(env)

    self.observation_space = gym.spaces.Box(
        low=0,
        high=255,
        shape=(self.env.width * self.env.height * 3,),  # number of cells
        dtype='float32')

  def observation(self, obs):
    env = self.unwrapped
    full_grid = env.grid.encode()
    full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array(
        [OBJECT_TO_IDX['agent'], COLOR_TO_IDX['red'], env.agent_dir])
    full_grid = full_grid.astype(np.float32)
    full_grid[:, :, 0] /= (len(OBJECT_TO_IDX) - 1)
    full_grid[:, :, 1] /= (len(COLOR_TO_IDX) - 1)
    full_grid[:, :, 2] /= 3
    return full_grid.flatten()


class FullyObsWrapperV4(gym.core.ObservationWrapper):
  """Fully observable gridworld using a compact grid encoding.

  Normalizes output.
  """

  def __init__(self, env):
    super().__init__(env)

    self.observation_space = gym.spaces.Box(
        low=0,
        high=255,
        shape=(self.env.width * self.env.height * 3,),  # number of cells
        dtype='float32')

  def observation(self, obs):
    env = self.unwrapped
    full_grid = env.grid.encode()
    full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array(
        [OBJECT_TO_IDX['agent'], COLOR_TO_IDX['red'], env.agent_dir])
    full_grid = full_grid.astype(np.float32)
    full_grid[:, :, 0] /= len(OBJECT_TO_IDX)
    full_grid[:, :, 1] /= len(COLOR_TO_IDX)
    full_grid[:, :, 2] /= 4
    return full_grid.flatten()


class ProcGenObsWrapper(gym.core.ObservationWrapper):
  """Wraps procgen obs env."""

  def __init__(self, env):
    super().__init__(env)
    self.img_size = (64, 64, 3)
    # img_size = functools.reduce(operator.mul, new_img_size, 1)

    self.observation_space = gym.spaces.Box(
        low=0,
        high=1.0,
        shape=self.img_size,  # number of cells
        dtype='float32')

  def observation(self, obs):
    obs = np.array(
        Image.fromarray(obs).resize(self.img_size[:2], Image.BILINEAR),
        dtype=np.float32)
    # obs = tf.image.resize(obs, (self.img_size[:2])).numpy()
    obs = obs / 255.
    return obs


def make_environment_bsuite(bsuite_id='catch/0'):
  # raw_environment = bsuite.load_and_record_to_csv(
  #    bsuite_id=bsuite_id,
  #    results_dir=results_dir,
  #   overwrite=True)
  raw_environment = bsuite.load_from_id(bsuite_id)
  environment = wrappers.SinglePrecisionWrapper(raw_environment)
  return environment


def make_environment_atari(env_id, evaluation: bool = False):
  import atari_py  # pylint: disable=unused-import, g-import-not-at-top
  flags.FLAGS.atari_roms_path = '/tmp/atari_roms/'
  env = gym.make(env_id, full_action_space=True)

  max_episode_len = 108_000 if evaluation else 50_000

  return wrappers.wrap_all(env, [
      wrappers.GymAtariAdapter,
      functools.partial(
          wrappers.AtariWrapper,
          to_float=True,
          max_episode_len=max_episode_len,
          zero_discount_on_life_loss=True,
      ),
      wrappers.SinglePrecisionWrapper,
  ])


class StateBonus(gym.core.Wrapper):
  """Exploration wrapper.

  Adds an exploration bonus based on which positions are visited on the grid.
  """

  def __init__(self, env):
    super().__init__(env)
    self.counts = {}

  def step(self, action):
    obs, reward, done, info = self.env.step(action)

    # Tuple based on which we index the counts
    # We use the position after an update
    env = self.unwrapped
    tup = (tuple(env.agent_pos))

    # Get the count for this key
    pre_count = 0
    if tup in self.counts:
      pre_count = self.counts[tup]

    # Update the count for this key
    new_count = pre_count + 1
    self.counts[tup] = new_count

    bonus = 1 / math.sqrt(new_count)
    reward += bonus

    return obs, reward, done, info

  def reset(self, **kwargs):
    return self.env.reset(**kwargs)


def check_env_id_bsuite(env_id):
  return not env_id[0].isupper()


def make_environment_gym(
    env_id: str,
    minigrid_obs_version: str = 'v1',
    evaluation: bool = False,
    seed: Optional[int] = None,
) ->...:
  """Makes a deepmind environment and wraps it."""
  if 'MiniGrid' in env_id:
    reward_wrapper = FloatRewardWrapper
    if 'ScaleR10000' in env_id:
      env_id = env_id.replace('ScaleR10000', '')
      reward_wrapper = ScaleR10000Wrapper
    if 'ScaleR1000' in env_id:
      env_id = env_id.replace('ScaleR1000', '')
      reward_wrapper = ScaleR1000Wrapper
    if 'ScaleR100' in env_id:
      env_id = env_id.replace('ScaleR100', '')
      reward_wrapper = ScaleR100Wrapper
    if 'ScaleR10' in env_id:
      env_id = env_id.replace('ScaleR10', '')
      reward_wrapper = ScaleR10Wrapper
    env = gym.make(env_id)
    if minigrid_obs_version == 'v1':
      env = FullyObsWrapper(env)
    elif minigrid_obs_version == 'v2':
      env = FullyObsWrapperV2(env)
    elif minigrid_obs_version == 'v3':
      env = FullyObsWrapperV3(env)
    elif minigrid_obs_version == 'v4':
      env = FullyObsWrapper(env)
      env = StateBonus(env)
    elif minigrid_obs_version == 'v5':
      env = FlatObsWrapper(env)
    elif minigrid_obs_version == 'v6':
      env = OneHotPartialObsWrapper(env)
    elif minigrid_obs_version == 'v7':
      env = FullyObsWrapperV4(env)
    env = reward_wrapper(env)
    env = wrappers.gym_wrapper.GymWrapper(env)
    env = wrappers.SinglePrecisionWrapper(env)
  elif env_id in atari_env_config.keys():
    env = make_environment_atari(env_id, evaluation=evaluation)
  elif check_env_id_bsuite(env_id):
    env = make_environment_bsuite(env_id)
  else:
    env = wrappers.gym_wrapper.GymWrapper(FloatRewardWrapper(gym.make(env_id)))
    env = wrappers.SinglePrecisionWrapper(env)
  if seed is not None:
    env.seed(seed)
  return env
