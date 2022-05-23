"""Epsilon-greedy scheduling libraries."""

import abc
import dataclasses
from typing import Optional

from acme import types
from acme.agents.tf import actors
from acme.tf import utils as tf2_utils
from acme.utils import counting
import tensorflow as tf
import trfl


class Schedule(abc.ABC):

  @abc.abstractmethod
  def value(self, step: int) -> float:
    """Returns scheduled value based on step count."""


@dataclasses.dataclass
class LinearSchedule(Schedule):
  init_value: float
  final_value: float
  scheduled_steps: float

  def value(self, step: int) -> float:
    fraction = min(step / self.scheduled_steps, 1.0)
    return self.init_value + fraction * (self.final_value - self.init_value)


@dataclasses.dataclass
class AtariEpsilonSchedule(Schedule):
  init_value: float
  scheduled_steps: float = 25e6

  def value(self, step: int) -> float:
    exponent = 1 + 7 * min(step / self.scheduled_steps, 1)
    return self.init_value**exponent


class FeedForwardActorWithEpsilonSchedule(actors.FeedForwardActor):
  """A feed-forward actor with epislon-greedy scheduling.

  Scheduling steps are maintained in a `counting.Counter` object, which is
  convenient for distributed training and checkpointing.
  """

  def __init__(self, schedule: Schedule, counter: Optional[counting.Counter],
               **kwargs):
    self._schedule = schedule
    self._counter = counting.Counter(counter)
    self._epsilon = tf.Variable(0.0, trainable=False)
    super().__init__(**kwargs)

  @tf.function
  def _policy(self, observation: types.NestedTensor) -> types.NestedTensor:
    # Add a dummy batch dimension and as a side effect convert numpy to TF.
    batched_observation = tf2_utils.add_batch_dim(observation)

    # Compute the policy, conditioned on the observation.
    policy = self._policy_network(batched_observation)
    # If we have a tuple (e.g. in C51 agents), assume the first entry represents
    # Q-values.
    if isinstance(policy, tuple):
      policy = policy[0]
    policy_output = trfl.epsilon_greedy(policy, epsilon=self._epsilon)
    return policy_output.sample()

  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
    if self._schedule is not None:
      epsilon = self._schedule.value(
          step=self._counter.get_counts().get('epsilon_schedule_steps', 0))
    else:
      epsilon = 0.0
    self._epsilon.assign(epsilon)

    policy_output = self._policy(observation)
    action = tf2_utils.to_numpy_squeeze(policy_output)
    self._counter.increment(epsilon_schedule_steps=1)
    return action
