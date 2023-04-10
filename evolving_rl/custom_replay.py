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

"""An in-memory lightweight trajectory-based replay buffer."""
import dataclasses
from typing import Dict, Iterator, Optional, Tuple

from acme import adders
from acme import specs
from acme import types
import dm_env
import numpy as np
import reverb
import tensorflow as tf


@dataclasses.dataclass
class TensorSpec:
  shape: Tuple[int, ...]
  dtype: np.dtype


ExtrasSpecs = Dict[str, TensorSpec]


class TransitionReplayLite(adders.Adder, Iterator[reverb.ReplaySample]):
  """An in-memory single step replay buffer for offline RL algos (SAC).

  Stores single step transitions instead of trajectories.

  It inherits two base classes:
    1) As an `adders.Adder`, it can be referenced by an actor for inserting
       experiences.
    2) As an `Iterator[reverb.ReplaySample]`, it can be referenced by a learner
       for sampling.

  This can replace the reverb replay server and client, and is compatible with
  the ActorLearnerBuilder interface.

  NOTE: This is highly specialized for our use case for meta-pg learning. It has
  a few assumptions:
    1) Observation, action, reward, discount specs are all flat (i.e. not
       nested). This is convenient for constructing the in-memory buffer.
    2) We do not implement reset() and signature() methods as they are not
       needed.
    3) We don't support a few concepts in reverb (e.g. info). They are not used.
  """

  def __init__(
      self,
      env_spec: specs.EnvironmentSpec,
      minibatch_size: int,
      buffer_size: int,
      extras_specs: Optional[ExtrasSpecs] = None,
  ):
    """Init.

    Args:
      env_spec: Environment spec.
      minibatch_size: The buffer will sample minibatches of this size
      buffer_size: The buffer will hold this many steps
      extras_specs: Specs for extras. If None, a default is provided for
        log_prob, which is typical for policy gradient algorithms.
    """
    self._rng = np.random.default_rng(seed=1)
    self._buffer = None
    self._minibatch_size = minibatch_size
    self._capacity = buffer_size
    self._env_spec = env_spec
    self._extras_specs = extras_specs or {
        "log_prob": TensorSpec((), np.float32)  # pytype: disable=wrong-arg-types  # numpy-scalars
    }

    # Init states
    self._observations = np.zeros(
        (buffer_size,) + self._env_spec.observations.shape,
        self._env_spec.observations.dtype)
    self._actions = np.zeros((buffer_size,) + self._env_spec.actions.shape,
                             self._env_spec.actions.dtype)
    self._rewards = np.zeros((buffer_size, 1), np.float32)
    self._discount = np.zeros((buffer_size, 1), np.float32)
    self._terminal = np.zeros((buffer_size, 1), bool)
    self._extras = {
        k: np.zeros((buffer_size,) + v.shape, v.dtype)
        for k, v in self._extras_specs.items()
    }

    # We maintain these buffers in the following structure:
    # s_t, s_tp1, ...
    # a_t, ...
    # r_t, ...
    # d_t, ...
    # ..., terminal_t, ...
    #
    # NOTE: terminal_t is aligned with s_tp1. During sampling, we only sample
    # "non-terminal columns" and "non-leading columns".
    # Index of the "leading column"
    self._step_idx = -1
    # Number of states added to buffer so far. Capped at capacity.
    self._size = 0
    self._add_first_called = False
    self._valid_mask = np.zeros((self._capacity, 1), dtype=bool)

  def get_size(self):
    return self._size

  def _reset_buffer(self):
    self._size = 0
    self._step_idx = -1
    self._valid_mask = np.zeros((self._capacity, 1), dtype=bool)

  def add_first(self, timestep: dm_env.TimeStep):
    """See base class `adders.Adder`."""
    self._add_first_called = True
    self._size = min(self._size + 1, self._capacity)
    self._step_idx = (self._step_idx + 1) % self._capacity
    self._observations[self._step_idx] = timestep.observation
    # Leading state cannot be sampled.
    self._valid_mask[self._step_idx, 0] = 0

  def add(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
      extras: types.NestedArray = (),
  ):
    """See base class `adders.Adder`."""
    assert self._add_first_called, "add_first() not called yet!"
    t = self._step_idx
    tp1 = (t + 1) % self._capacity
    self._valid_mask[t, 0] = 1
    self._valid_mask[tp1, 0] = 0  # Leading state cannot be sampled
    self._actions[t] = action
    self._observations[tp1] = next_timestep.observation
    self._rewards[t] = next_timestep.reward
    self._discount[t] = next_timestep.discount

    # Terminal will be 1 at same idx as terminal state
    self._terminal[tp1] = next_timestep.last()
    if extras:
      for key in extras:
        self._extras[key][t] = extras[key]
    if next_timestep.last():
      # Padding zero. During sampling, only positions w/ non-terminals
      # states are sampled. So these dummy values are never used.
      self._actions[tp1] = np.zeros(self._env_spec.actions.shape,
                                    self._env_spec.actions.dtype)
      self._rewards[tp1] = np.zeros(self._env_spec.rewards.shape,
                                    np.float32)
      self._discount[tp1] = np.zeros(self._env_spec.discounts.shape,
                                     bool)
      self._add_first_called = False

    self._size = min(self._size + 1, self._capacity)
    self._step_idx = tp1

  def reset(self):
    """Unused method from `adders.Adder`."""
    raise NotImplementedError

  @classmethod
  def signature(cls,
                environment_spec: specs.EnvironmentSpec,
                extras_spec: types.NestedArray,
                sequence_length: Optional[int] = None):
    """Unused method from `adders.Adder`."""
    raise NotImplementedError

  def __iter__(self):
    """Implements iterator interface."""
    return self

  def __next__(self):
    """Sample random batch from buffer."""
    n = min(self._size, self._capacity)
    idx = self._rng.choice(
        np.nonzero(self._valid_mask[:n])[0],
        size=self._minibatch_size,
        replace=False)
    idxp1 = (idx + 1) % self._capacity
    extras = {k: self._extras[k][idx] for k in self._extras.keys()}
    data = types.Transition(
        observation=tf.convert_to_tensor(self._observations[idx]),
        action=tf.convert_to_tensor(self._actions[idx]),
        reward=tf.convert_to_tensor(self._rewards[idx]),
        discount=tf.convert_to_tensor(self._discount[idx]),
        next_observation=tf.convert_to_tensor(self._observations[idxp1]),
        extras=extras,
    )
    # Set a dummy info object. We don't use it.
    info = reverb.SampleInfo(*[0 for _ in reverb.SampleInfo.tf_dtypes()])
    return reverb.ReplaySample(info=info, data=data)
