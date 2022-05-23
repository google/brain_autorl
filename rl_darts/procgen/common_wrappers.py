"""Common wrappers for environments."""
import gym
import numpy as np


class FloatRewardWrapper(gym.core.RewardWrapper):
  """Wraps the reward to float, which is standard for `dm_env.Environment`."""

  def reward(self, reward):
    return float(reward)


class ObsToFloat(gym.ObservationWrapper):
  """Converts observations to floating point values, and changes observation space to dtype float.

  Also can scale values.
  """

  def __init__(self, env, divisor=1.0):
    self.env = env
    self.divisor = divisor
    self.action_space = self.env.action_space
    self.observation_space = self.env.observation_space
    self.metadata = self.env.metadata
    self.observation_space.dtype = np.float32

  def observation(self, observation):
    if isinstance(observation, dict):
      # dict space
      scalar_obs = {}
      for k, v in observation.items():
        scalar_obs[k] = v.astype(np.float32)
      return scalar_obs / self.divisor
    else:
      return observation.astype(np.float32) / self.divisor
