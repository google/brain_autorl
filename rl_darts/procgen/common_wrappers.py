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
