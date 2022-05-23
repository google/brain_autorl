"""Custom variants of TF-Agent Networks, which allow any feature network.

Modified from tf_agents.networks. Contains networks for both PPO
and DQN settings.

NOTE: For keras Layer API, modules must be attached to `self` AFTER
super().__init__(...) for correct variable capturing.
"""
from typing import Optional, TypeVar, Tuple

import sonnet.v2 as snt
import tensorflow as tf
from tf_agents.keras_layers import dynamic_unroll_layer
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import network
from tf_agents.networks import q_network
from tf_agents.networks import utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step
from tf_agents.utils import nest_utils

FeedForwardType = TypeVar("FeedForwardType", bound=snt.Module)
SonnetRNNCellType = TypeVar("SonnetRNNCellType", bound=snt.RNNCore)
KerasRNNCellType = TypeVar("KerasRNNCellType", bound=tf.keras.layers.Layer)


class SonnetRNNCelltoKerasRNNCell(tf.keras.layers.Layer):
  """Converts a snt.RNNCore into a RNNCell under tf.keras.Layer API."""

  def __init__(self, sonnet_rnn: SonnetRNNCellType, **kwargs):
    output_size = sonnet_rnn._hidden_size
    state_size = [sonnet_rnn.initial_state(batch_size=1).shape[-1]]

    super().__init__(**kwargs)

    self.output_size = output_size
    self.state_size = state_size
    self._sonnet_rnn = sonnet_rnn

  def build(self, input_shape):
    # Sonnet module already built!
    self.built = True

  def call(self, inputs, states, training=None):
    output, rnn_state = self._sonnet_rnn(inputs=inputs, prev_state=states[0])
    return output, [rnn_state]

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    return [self._sonnet_rnn.initial_state(batch_size=batch_size)]

  @property
  def trainable_weights(self):
    return self._sonnet_rnn.trainable_variables

  @property
  def non_trainable_weights(self):
    trainable_variables_names = [
        v.name for v in self._sonnet_rnn.trainable_variables
    ]
    all_variables = self._sonnet_rnn.variables
    non_trainable_variables = []
    for var in all_variables:
      if var.name not in trainable_variables_names:
        non_trainable_variables.append(var)
    return tuple(non_trainable_variables)


class CustomEncodingNetwork(network.Network):
  """Custom Feature Network. Feature network must flatten final output."""

  def __init__(self,
               input_tensor_spec,
               feature_network: FeedForwardType,
               rnn_cell: Optional[KerasRNNCellType] = None,
               batch_squash=True,
               dtype=tf.float32,
               name="EncodingNetwork"):

    if rnn_cell:
      rnn_network = dynamic_unroll_layer.DynamicUnroll(rnn_cell)

      def create_spec(size):
        return tensor_spec.TensorSpec(size, dtype=dtype)

      state_spec = tf.nest.map_structure(create_spec,
                                         rnn_network.cell.state_size)
    else:
      rnn_network = None
      state_spec = ()

    super(CustomEncodingNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec, state_spec=state_spec, name=name)

    self._feature_network = feature_network
    self._rnn_network = rnn_network
    self._batch_squash = batch_squash

  def call(self, observation, step_type=None, network_state=(), training=False):
    if self._rnn_network:
      num_outer_dims = nest_utils.get_outer_rank(observation,
                                                 self.input_tensor_spec)
      if num_outer_dims not in (1, 2):
        raise ValueError(
            "Input observation must have a batch or batch x time outer shape.")

      has_time_dim = num_outer_dims == 2
      if not has_time_dim:
        # Add a time dimension to the inputs.
        observation = tf.nest.map_structure(lambda t: tf.expand_dims(t, 1),
                                            observation)
        step_type = tf.nest.map_structure(lambda t: tf.expand_dims(t, 1),
                                          step_type)

    features, _ = self._feature_network_call(
        observation=observation,
        step_type=step_type,
        network_state=network_state,
        training=training)

    if self._rnn_network:
      output, rnn_state = self._rnn_network_call(features, step_type,
                                                 network_state, training,
                                                 has_time_dim)
      return output, rnn_state
    else:
      return features, network_state

  def _feature_network_call(self, observation, step_type,
                            network_state: Tuple[tf.Tensor], training: bool):
    del step_type  # unused.

    if self._batch_squash:
      outer_rank = nest_utils.get_outer_rank(observation,
                                             self.input_tensor_spec)
      batch_squash = utils.BatchSquash(outer_rank)
      observation = tf.nest.map_structure(batch_squash.flatten, observation)

    features = self._feature_network(observation, is_training=training)

    if self._batch_squash:
      features = tf.nest.map_structure(batch_squash.unflatten, features)

    return features, network_state

  def _rnn_network_call(self, rnn_input, step_type, rnn_state, training: bool,
                        has_time_dim: bool):
    network_kwargs = {}
    network_kwargs["reset_mask"] = tf.equal(
        step_type, time_step.StepType.FIRST, name="mask")

    output, final_rnn_state = self._rnn_network(
        inputs=rnn_input,
        initial_state=rnn_state,
        training=training,
        **network_kwargs)

    if not has_time_dim:
      # Remove time dimension from the output.
      output = tf.squeeze(output, [1])

    return output, final_rnn_state


class CustomActorDistributionNetwork(network.DistributionNetwork):
  """Processes output of the feature network for action distribution."""

  def __init__(self,
               output_tensor_spec,
               encoder: CustomEncodingNetwork,
               discrete_projection_net=actor_distribution_network
               ._categorical_projection_net,
               continuous_projection_net=actor_distribution_network
               ._normal_projection_net,
               name="ActorDistributionNetwork"):

    def map_proj(spec):
      if tensor_spec.is_discrete(spec):
        return discrete_projection_net(spec)
      else:
        return continuous_projection_net(spec)

    projection_networks = tf.nest.map_structure(map_proj, output_tensor_spec)
    output_spec = tf.nest.map_structure(lambda proj_net: proj_net.output_spec,
                                        projection_networks)

    super(CustomActorDistributionNetwork, self).__init__(
        input_tensor_spec=encoder.input_tensor_spec,
        state_spec=encoder.state_spec,
        output_spec=output_spec,
        name=name)

    self._encoder = encoder
    self._projection_networks = projection_networks
    self._output_tensor_spec = output_tensor_spec

  @property
  def output_tensor_spec(self):
    return self._output_tensor_spec

  def call(self,
           observation,
           step_type,
           network_state,
           training=False,
           mask=None):
    output, network_state = self._encoder(
        observation, step_type, network_state, training=training)

    outer_rank = nest_utils.get_outer_rank(observation, self.input_tensor_spec)

    output_actions = tf.nest.map_structure(
        lambda proj_net: proj_net(output, outer_rank, training=training)[0],
        self._projection_networks)

    return output_actions, network_state


class CustomValueNetwork(network.Network):
  """Feed Forward value network after feature network.

  Reduces to 1 value output per batch item.
  """

  def __init__(self, encoder: CustomEncodingNetwork, name="ValueNetwork"):

    postprocessing_layers = tf.keras.layers.Dense(
        1,
        activation=None,
        kernel_initializer=tf.random_uniform_initializer(
            minval=-0.03, maxval=0.03))

    super().__init__(
        input_tensor_spec=encoder.input_tensor_spec,
        state_spec=encoder.state_spec,
        name=name)

    self._encoder = encoder
    self._postprocessing_layers = postprocessing_layers

  def call(self, observation, step_type=None, network_state=(), training=False):
    output, network_state = self._encoder(
        observation, step_type, network_state, training=training)
    value = self._postprocessing_layers(output, training=training)
    return tf.squeeze(value, -1), network_state


class CustomQNetwork(network.Network):
  """Feed Forward network."""

  def __init__(self,
               input_tensor_spec,
               action_spec,
               feature_network,
               batch_squash=True,
               dtype=tf.float32,
               name="QNetwork"):
    """Creates an instance of `QNetwork`.

    Args:
      input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
        input observations.
      action_spec: A nest of `tensor_spec.BoundedTensorSpec` representing the
        actions.
      feature_network: Custom encoding network.
      batch_squash: If True the outer_ranks of the observation are squashed into
        the batch dimension. This allow encoding networks to be used with
        observations with shape [BxTx...].
      dtype: The dtype to use by the convolution and fully connected layers.
      name: A string representing the name of the network.

    Raises:
      ValueError: If `input_tensor_spec` contains more than one observation. Or
        if `action_spec` contains more than one action.
    """
    q_network.validate_specs(action_spec, input_tensor_spec)
    action_spec = tf.nest.flatten(action_spec)[0]
    num_actions = action_spec.maximum - action_spec.minimum + 1

    encoder = CustomEncodingNetwork(
        input_tensor_spec,
        feature_network,
        batch_squash=batch_squash,
        dtype=dtype)

    q_value_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.random_uniform_initializer(
            minval=-0.03, maxval=0.03),
        bias_initializer=tf.constant_initializer(-0.2),
        dtype=dtype)

    super(CustomQNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec, state_spec=(), name=name)

    self._encoder = encoder
    self._q_value_layer = q_value_layer

  def call(self, observation, step_type=None, network_state=(), training=False):
    """Runs the given observation through the network.

    Args:
      observation: The observation to provide to the network.
      step_type: The step type for the given observation. See `StepType` in
        time_step.py.
      network_state: A state tuple to pass to the network, mainly used by RNNs.
      training: Whether the output is being used for training.

    Returns:
      A tuple `(logits, network_state)`.
    """
    state, network_state = self._encoder(
        observation,
        step_type=step_type,
        network_state=network_state,
        training=training)
    q_value = self._q_value_layer(state, training=training)
    return q_value, network_state


class CustomCategoricalQNetwork(network.Network):
  """Creates a categorical Q-network.

  It can be used to take an input of batched observations and outputs
  ([batch_size, num_actions, num_atoms], network's state).

  The first element of the output is a batch of logits based on the distribution
  called C51 from Bellemare et al., 2017 (https://arxiv.org/abs/1707.06887). The
  logits are used to compute approximate probability distributions for Q-values
  for each potential action, by computing the probabilities at the 51 points
  (called atoms) in np.linspace(-10.0, 10.0, 51).
  """

  def __init__(self,
               input_tensor_spec,
               action_spec,
               feature_network,
               num_atoms=51,
               name="CategoricalQNetwork"):
    """Creates an instance of `CategoricalQNetwork`.

    The logits output by __call__ will ultimately have a shape of
    `[batch_size, num_actions, num_atoms]`, where `num_actions` is computed as
    `action_spec.maximum - action_spec.minimum + 1`. Each value is a logit for
    a particular action at a particular atom (see above).

    As an example, if
    `action_spec = tensor_spec.BoundedTensorSpec([1], tf.int32, 0, 4)` and
    `num_atoms = 51`, the logits will have a shape of `[batch_size, 5, 51]`.

    Args:
      input_tensor_spec: A `tensor_spec.TensorSpec` specifying the observation
        spec.
      action_spec: A `tensor_spec.BoundedTensorSpec` representing the actions.
      feature_network: Custom feature encoding network.
      num_atoms: The number of atoms to use in our approximate probability
        distributions. Defaults to 51 to produce C51.
      name: A string representing the name of the network.

    Raises:
      TypeError: `action_spec` is not a `BoundedTensorSpec`.
    """
    super(CustomCategoricalQNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec, state_spec=(), name=name)

    if not isinstance(action_spec, tensor_spec.BoundedTensorSpec):
      raise TypeError("action_spec must be a BoundedTensorSpec. Got: %s" %
                      (action_spec,))

    self._num_actions = action_spec.maximum - action_spec.minimum + 1
    self._num_atoms = num_atoms

    q_network_action_spec = tensor_spec.BoundedTensorSpec(
        (), tf.int32, minimum=0, maximum=self._num_actions * num_atoms - 1)

    self._q_network = CustomQNetwork(
        input_tensor_spec=input_tensor_spec,
        action_spec=q_network_action_spec,
        feature_network=feature_network,
        name=name)

  @property
  def num_atoms(self):
    return self._num_atoms

  def call(self, observation, step_type=None, network_state=(), training=False):
    """Runs the given observation through the network.

    Args:
      observation: The observation to provide to the network.
      step_type: The step type for the given observation. See `StepType` in
        time_step.py.
      network_state: A state tuple to pass to the network, mainly used by RNNs.
      training: Whether the output will be used for training.

    Returns:
      A tuple `(logits, network_state)`.
    """
    logits, network_state = self._q_network(
        observation, step_type, network_state, training=training)
    logits = tf.reshape(logits, [-1, self._num_actions, self._num_atoms])
    return logits, network_state
