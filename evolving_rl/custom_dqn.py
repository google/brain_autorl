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

"""Custom DQN from Acme which supports a custom loss function.

Agent and learner will take a custom loss function as part of the class
initialization.
"""
# pytype: skip-file
import copy
import time
from typing import Dict, Optional

from acme import adders
from acme import core
from acme import datasets
from acme import specs
from acme import types
from acme.adders import reverb as rvadders
from acme.agents import agent
from acme.agents.tf.dqn import DQNLearner
from acme.tf import networks
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils
from acme.utils import counting
from acme.utils import loggers
from brain_autorl.evolving_rl.custom_replay import TransitionReplayLite
from brain_autorl.evolving_rl.env_config import atari_env_config
from brain_autorl.evolving_rl.env_utils import LinearSchedule
import dm_env
import reverb
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
import tree
import trfl
from trfl import indexing_ops

tfd = tfp.distributions


def make_networks(env_id, env_spec, mlp_size):
  if env_id in atari_env_config.keys():
    network = networks.DQNAtariNetwork(env_spec.actions.num_values)
  else:
    network = snt.Sequential([
        snt.Flatten(),
        snt.nets.MLP([mlp_size, mlp_size, env_spec.actions.num_values])
    ])
  return {'network': network}


class CustomDQNLearner(DQNLearner):
  """DQN learner.

  This is the learning component of a DQN agent. It takes a dataset as input
  and implements update functionality to learn from this dataset. Optionally
  it takes a replay client as well to allow for updating of priorities.
  """

  def __init__(self,
               network: snt.Module,
               target_network: snt.Module,
               discount: float,
               importance_sampling_exponent: float,
               learning_rate: float,
               target_update_period: int,
               dataset: tf.data.Dataset,
               huber_loss_parameter: float = 1.,
               replay_client: reverb.TFClient = None,
               counter: counting.Counter = None,
               logger: loggers.Logger = None,
               checkpoint: bool = True,
               use_priority: bool = False,
               loss_program=None,
               reward_scale=1.0):
    """Initializes the learner.

    Args:
      network: the online Q network (the one being optimized)
      target_network: the target Q critic (which lags behind the online net).
      discount: discount to use for TD updates.
      importance_sampling_exponent: power to which importance weights are raised
        before normalizing.
      learning_rate: learning rate for the q-network update.
      target_update_period: number of learner steps to perform before updating
        the target networks.
      dataset: dataset to learn from, whether fixed or from a replay buffer (see
        `acme.datasets.reverb.make_reverb_dataset` documentation).
      huber_loss_parameter: Quadratic-linear boundary for Huber loss.
      replay_client: client to replay to allow for updating priorities.
      counter: Counter object for (potentially distributed) counting.
      logger: Logger object for writing logs to.
      checkpoint: boolean indicating whether to checkpoint the learner.
      use_priority: Whether to use priority replay or not.
      loss_program: Loss program to compute loss function.
      reward_scale: Scale on the reward function.
    """
    self.reward_scale = reward_scale
    self.use_priority = use_priority
    DQNLearner.__init__(
        self,
        network=network,
        target_network=target_network,
        discount=discount,
        importance_sampling_exponent=importance_sampling_exponent,
        learning_rate=learning_rate,
        target_update_period=target_update_period,
        dataset=dataset,
        huber_loss_parameter=huber_loss_parameter,
        replay_client=replay_client,
        counter=counter,
        logger=logger,
        checkpoint=checkpoint,)
    self._loss_program = loss_program

  @tf.function
  def _step(self, inputs) -> Dict[str, tf.Tensor]:
    """Do a step of SGD and update the priorities."""
    transitions = inputs.data
    o_tm1, a_tm1, r_t, d_t, o_t = (transitions.observation, transitions.action,
                                   transitions.reward, transitions.discount,
                                   transitions.next_observation)

    dtype = tf.float32
    with tf.GradientTape() as tape:
      # Evaluate our networks.
      # The rewards and discounts have to have the same type as network values.
      r_t = tf.cast(r_t, dtype)
      # Replace nans with 0
      r_t = tf.where(tf.math.is_nan(r_t), tf.zeros_like(r_t), r_t)
      if self.reward_scale != 1.0:
        r_t /= self.reward_scale
      d_t = tf.cast(d_t, dtype) * tf.cast(self._discount, dtype)

      # Make sure that floats r_t and d_t have 1 in last dimension
      # so that broadcasting works correctly.
      r_t = tf.reshape(r_t, (-1, 1))
      d_t = tf.reshape(d_t, (-1, 1))

      loss = self._loss_program(self._network, a_tm1, o_tm1,
                                self._target_network, o_t, r_t, d_t)
      loss = tf.reduce_mean(loss, axis=[0])  # []

    # Do a step of SGD.
    gradients = tape.gradient(loss, self._network.trainable_variables)
    self._optimizer.apply(gradients, self._network.trainable_variables)

    # Update the priorities in the replay buffer.
    if self._replay_client and self.use_priority:
      keys, _ = inputs.info[:2]
      r_t = tf.squeeze(r_t)
      d_t = tf.squeeze(d_t)

      qsa = indexing_ops.batched_index(tf.squeeze(self._network(o_tm1)), a_tm1)
      qtarg_list = self._target_network(o_t)
      targ = r_t + d_t * tf.reduce_max(qtarg_list, axis=1)
      td = 0.5 * tf.math.square(targ - qsa)
      priorities = tf.cast(td, tf.float64)

      self._replay_client.update_priorities(
          table=rvadders.DEFAULT_PRIORITY_TABLE,
          keys=keys,
          priorities=priorities)

    # Periodically update the target network.
    if tf.math.mod(self._num_steps, self._target_update_period) == 0:
      for src, dest in zip(self._network.variables,
                           self._target_network.variables):
        dest.assign(src)
    self._num_steps.assign_add(1)

    # Report loss & statistics for logging.
    fetches = {
        'loss': loss,
    }

    return fetches

  def step(self):
    # Pull out the data needed for updates/priorities.
    inputs = next(self._iterator)
    # Do a batch of SGD.
    result = self._step(inputs)

    # Compute elapsed time.
    timestamp = time.time()
    elapsed_time = timestamp - self._timestamp if self._timestamp else 0
    self._timestamp = timestamp

    # Update our counts and record it.
    counts = self._counter.increment(steps=1, walltime=elapsed_time)
    result.update(counts)

    # Snapshot and attempt to write logs.
    if self._snapshotter is not None:
      self._snapshotter.save()
    self._logger.write(result)

  @tf.function
  def get_info(self) -> Dict[str, tf.Tensor]:
    """Log internal values of the loss function for debugging.

    This function is used for logging internal values such as the q values or
    different cases of the loss function. It will be called in the environment
    training loop if record_qval is set to True.

    Returns:
    Dictionary of values to log.
    """

    # Pull out the data needed for updates/priorities.
    inputs = next(self._iterator)
    transitions = inputs.data
    o_tm1, a_tm1, r_t, d_t, o_t = (transitions.observation, transitions.action,
                                   transitions.reward, transitions.discount,
                                   transitions.next_observation)
    dtype = tf.float32

    # Evaluate our networks.
    # The rewards and discounts have to have the same type as network values.
    r_t = tf.cast(r_t, dtype)
    # Replace nans with 0
    r_t = tf.where(tf.math.is_nan(r_t), tf.zeros_like(r_t), r_t)
    d_t = tf.cast(d_t, dtype) * tf.cast(self._discount, dtype)

    # Val1
    qsa = indexing_ops.batched_index(tf.squeeze(self._network(o_tm1)), a_tm1)
    qtarg_list = self._target_network(o_t)
    targ = r_t + d_t * tf.reduce_max(qtarg_list, axis=1)
    td = tf.math.square(targ - qsa)
    val1 = qsa
    val2 = td + targ
    val3 = td - targ
    val4 = tf.math.square(targ) * d_t

    case1 = tf.reduce_mean(
        tf.cast(
            tf.math.logical_and(
                tf.math.greater(val1, val2), tf.math.greater(val3, val4)),
            tf.float32))
    case2 = tf.reduce_mean(
        tf.cast(
            tf.math.logical_and(
                tf.math.greater(val1, val2), tf.math.less(val3, val4)),
            tf.float32))
    case3 = tf.reduce_mean(
        tf.cast(
            tf.math.logical_and(
                tf.math.less(val1, val2), tf.math.less(val3, val4)),
            tf.float32))
    case4 = tf.reduce_mean(
        tf.cast(
            tf.math.logical_and(
                tf.math.less(val1, val2), tf.math.greater(val3, val4)),
            tf.float32))

    # Report loss & statistics for logging.
    fetches = {
        'case1': case1,
        'case2': case2,
        'case3': case3,
        'case4': case4,
    }

    return fetches


class DQN(agent.Agent):
  """DQN agent.

  This implements a single-process DQN agent. This is a simple Q-learning
  algorithm that inserts N-step transitions into a replay buffer, and
  periodically updates its policy by sampling these transitions using
  prioritization. This is modified from the original implementation and supports
  a custom loss program, epislon schedule, and a toggle for using priority
  replay.
  """

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      network: snt.Module,
      batch_size: int = 256,
      prefetch_size: int = 4,
      target_update_period: int = 100,
      samples_per_insert: float = 32.0,
      min_replay_size: int = 1000,
      max_replay_size: int = 1000000,
      importance_sampling_exponent: float = 0.2,
      priority_exponent: float = 0.6,
      n_step: int = 5,
      epsilon_initial_p: float = 1.0,
      epsilon_final_p: float = 0.02,
      epsilon_schedule: int = 10000,
      learning_rate: float = 1e-3,
      discount: float = 0.99,
      logger: loggers.Logger = None,
      checkpoint: bool = True,
      use_priority: bool = False,
      loss_program=None,
      reward_scale: float = 1.0,
      use_reverb: bool = True,
      actor_sample_seed: Optional[int] = None,
  ):
    """Initializes the agent.

    Args:
      environment_spec: description of the actions, observations, etc.
      network: the online Q network (the one being optimized)
      batch_size: batch size for updates.
      prefetch_size: size to prefetch from replay.
      target_update_period: number of learner steps to perform before updating
        the target networks.
      samples_per_insert: number of samples to take from replay for every insert
        that is made.
      min_replay_size: minimum replay size before updating. This and all
        following arguments are related to dataset construction and will be
        ignored if a dataset argument is passed.
      max_replay_size: maximum replay size.
      importance_sampling_exponent: power to which importance weights are raised
        before normalizing.
      priority_exponent: exponent used in prioritized sampling.
      n_step: number of steps to squash into a single transition.
      epsilon_initial_p: probability of taking a random action; ignored if a
        policy network is given.
      epsilon_final_p: final epislon value at end of schedule.
      epsilon_schedule: number of steps until final epsilon value is reached.
      learning_rate: learning rate for the q-network update.
      discount: discount to use for TD updates.
      logger: logger object to be used by learner.
      checkpoint: boolean indicating whether to checkpoint the learner.
      use_priority: Whether to use priority replay or not.
      loss_program: Loss program to compute loss function.
      reward_scale: Scale on the reward function.
      use_reverb: If True, use reverb replay buffer; otherwise use a custom
        in-memory ring buffer.
      actor_sample_seed: See sample_seed in FeedForwardActor.
    """
    if use_reverb:
      # Create a replay server to add data to. This uses no limiter behavior in
      # order to allow the Agent interface to handle it.
      replay_table = reverb.Table(
          name=rvadders.DEFAULT_PRIORITY_TABLE,
          sampler=reverb.selectors.Prioritized(priority_exponent),
          remover=reverb.selectors.Fifo(),
          max_size=max_replay_size,
          rate_limiter=reverb.rate_limiters.MinSize(1),
          signature=rvadders.NStepTransitionAdder.signature(environment_spec))
      self._server = reverb.Server([replay_table], port=None)

      # The adder is used to insert observations into replay.
      address = f'localhost:{self._server.port}'
      adder = rvadders.NStepTransitionAdder(
          client=reverb.Client(address), n_step=n_step, discount=discount)

      # The dataset provides an interface to sample from replay.
      replay_client = reverb.TFClient(address)
      dataset = datasets.make_reverb_dataset(
          server_address=address,
          batch_size=batch_size,
          prefetch_size=prefetch_size)
    else:
      replay = TransitionReplayLite(
          environment_spec,
          minibatch_size=batch_size,
          buffer_size=max_replay_size,
      )
      dataset = replay  # The replay also implements the dataset interface.
      adder = replay  # The replay also implements the adder interface.
      replay_client = None

    policy_network = snt.Sequential([
        network,
    ])

    # Create a target network.
    target_network = copy.deepcopy(network)

    # Ensure that we create the variables before proceeding (maybe not needed).
    tf2_utils.create_variables(network, [environment_spec.observations])
    tf2_utils.create_variables(target_network, [environment_spec.observations])

    # Create an epsilon schedule that decays linearly.
    exploration = None
    if epsilon_initial_p > 0:
      exploration = LinearSchedule(
          schedule_timesteps=epsilon_schedule,
          initial_p=epsilon_initial_p,
          final_p=epsilon_final_p)
    # Create the actor which defines how we take actions.
    actor = FeedForwardActor(
        policy_network,
        adder=adder,
        exploration=exploration,
        sample_seed=actor_sample_seed,
    )

    # The learner updates the parameters (and initializes them).
    learner_cls = CustomDQNLearner
    learner = learner_cls(
        network=network,
        target_network=target_network,
        discount=discount,
        importance_sampling_exponent=importance_sampling_exponent,
        learning_rate=learning_rate,
        target_update_period=target_update_period,
        dataset=dataset,
        replay_client=replay_client,
        logger=logger,
        checkpoint=checkpoint,
        loss_program=loss_program,
        use_priority=use_priority,
        reward_scale=reward_scale,
    )

    if checkpoint:
      self._checkpointer = tf2_savers.Checkpointer(
          objects_to_save=learner.state,
          subdirectory='dqn_learner',
          time_delta_minutes=60.)
    else:
      self._checkpointer = None

    super().__init__(
        actor=actor,
        learner=learner,
        min_observations=max(batch_size, min_replay_size),
        observations_per_step=float(batch_size) / samples_per_insert)

  def update(self):
    super().update()
    if self._checkpointer is not None:
      self._checkpointer.save()


class FeedForwardActor(core.Actor):
  """A customized feed-forward actor which supports an epsilon schedule.

  An actor based on a feed-forward policy which takes non-batched observations
  and outputs non-batched actions. It also allows adding experiences to replay
  and updating the weights from the policy on the learner. This supports an
  epsilon schedule for the policy.
  """

  def __init__(
      self,
      policy_network: snt.Module,
      adder: Optional[adders.Adder] = None,
      exploration: LinearSchedule = None,
      variable_client: Optional[tf2_variable_utils.VariableClient] = None,
      sample_seed: Optional[int] = None,
  ):
    """Initializes the actor.

    Args:
      policy_network: the policy to run.
      adder: the adder object to which allows to add experiences to a
        dataset/replay buffer.
      exploration: A LinearSchedule object which determines the value of epsilon
        for the policy depending on how many steps have been taken.
      variable_client: object which allows to copy weights from the learner copy
        of the policy to the actor copy (in case they are separate).
      sample_seed: If provided, use it as a consistent seed for sampling from a
        tensorflow probability distribution. (This improves reproducibility.)
    """

    # Store these for later use.
    self._adder = adder
    self._variable_client = variable_client
    self._policy_network = tf.function(policy_network)
    self._exploration = exploration
    self._count = 0
    self._sample_seed = sample_seed

  def select_action(self, observation: types.NestedArray):
    # Add a dummy batch dimension and as a side effect convert numpy to TF.
    batched_obs = tf2_utils.add_batch_dim(observation)

    # Forward the policy network.
    policy_output = self._policy_network(batched_obs)

    qvals = tf2_utils.to_numpy_squeeze(policy_output)

    if self._exploration:
      epsilon = self._exploration.value(self._count)
      self._count += 1
    else:
      epsilon = 0.0

    policy_output = trfl.epsilon_greedy(policy_output, epsilon=epsilon)
    if self._sample_seed is not None:
      policy_output = policy_output.sample(seed=self._sample_seed)
    else:
      policy_output = policy_output.sample()

    # If the policy network parameterises a distribution, sample from it.
    def maybe_sample(output):
      if isinstance(output, tfd.Distribution):
        output = output.sample()
      return output

    policy_output = tree.map_structure(maybe_sample, policy_output)

    # Convert to numpy and squeeze out the batch dimension.
    action = tf2_utils.to_numpy_squeeze(policy_output)

    return action, qvals.tolist()

  def observe_first(self, timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add_first(timestep)

  def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add(action, next_timestep)

  def update(self):
    if self._variable_client:
      self._variable_client.update()
