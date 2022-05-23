"""Modified encoders from original PI-SAC codebase."""
from brain_autorl.rl_darts.policies import darts_cells
from brain_autorl.rl_darts.policies import darts_policies

import gin
import pyglove as pg
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.utils import nest_utils

tfkl = tf.keras.layers
tfkr = tf.keras.regularizers
tfd = tfp.distributions


@gin.configurable
class FRNConv(network.Network):
  """Convolutional encoder with FRN layers."""

  def __init__(
      self,
      input_tensor_spec,
      filters=(32, 32, 32, 32),
      strides=(2, 1, 1, 1),
      kernels=(3, 3, 3, 3),
      padding='valid',
      output_dim=50,
      output_tanh_activation=False,
      batch_squash=True,
      dtype=tf.float32,
      name='FRNConv',
      softmax_temperature=1.0,
  ):
    """Creates an instance of `FRNConv`."""
    super(FRNConv, self).__init__(
        input_tensor_spec=input_tensor_spec, state_spec=(), name=name)
    self.output_tanh_activation = output_tanh_activation
    self._batch_squash = batch_squash
    self._encoder = tf.keras.Sequential()
    self._uint8_input = input_tensor_spec.dtype == tf.uint8

    for i in range(len(filters)):
      self._encoder.add(
          tfkl.Conv2D(
              filters=filters[i],
              kernel_size=kernels[i],
              strides=strides[i],
              padding=padding,
              activation=None,
              dtype=dtype,
              name='%s/conv%d' % (name, i)))
      self._encoder.add(FRN())  # built-in TLU activation
    self._encoder.add(tfkl.Flatten())
    self._encoder.add(tfkl.Dense(output_dim, dtype=dtype, name='%s/fc' % name))
    self._encoder.add(tfkl.LayerNormalization(epsilon=1e-5))
    if self.output_tanh_activation:
      self._encoder.add(tfkl.Activation(tf.keras.activations.tanh))

  def call(self, inputs, step_type=None, network_state=(), training=False):
    del step_type  # unused.

    if self._uint8_input:
      inputs = tf.cast(inputs, tf.float32) / 255.00
    if self._batch_squash:
      outer_rank = nest_utils.get_outer_rank(inputs, self.input_tensor_spec)
      batch_squash = utils.BatchSquash(outer_rank)
      inputs = tf.nest.map_structure(batch_squash.flatten, inputs)
    states = inputs
    states = self._encoder(states, training=training)
    if self._batch_squash:
      states = tf.nest.map_structure(batch_squash.unflatten, states)
    return states, network_state


@gin.configurable
class FRNDARTSConv(network.Network):
  """DARTS Convolutional encoder with FRN layers."""

  def __init__(self,
               input_tensor_spec,
               filters=(32, 32, 32, 32),
               strides=(2, 1, 1, 1),
               kernels=(3, 3, 3, 3),
               padding='valid',
               output_dim=50,
               output_tanh_activation=False,
               batch_squash=True,
               dtype=tf.float32,
               name='FRNConv',
               cell_config_json=None,
               softmax_temperature=1.0):
    """Creates an instance of `FRNConv`."""
    super(FRNDARTSConv, self).__init__(
        input_tensor_spec=input_tensor_spec, state_spec=(), name=name)
    self.output_tanh_activation = output_tanh_activation
    self._batch_squash = batch_squash
    self._encoder = tf.keras.Sequential()
    self._uint8_input = input_tensor_spec.dtype == tf.uint8

    self.num_nodes = 4
    self.num_inputs = 1
    self.op_names = ['Conv3x3', 'Relu', 'Tanh', 'SkipConnection', 'Zero']
    self.default_output_mode = darts_cells.CellOutputMode.CONCAT_CONV
    self.softmax_temperature = softmax_temperature
    self.output_channels = 32
    if cell_config_json:
      normal_cell_config = pg.from_json_str(cell_config_json)
    else:
      normal_cell_config = darts_cells.DartsCellConfig.from_create_alpha(
          op_names=self.op_names,
          num_nodes=self.num_nodes,
          output_mode=self.default_output_mode,
          trainable=True,
          softmax_temperature=self.softmax_temperature)
    net_config = darts_policies.NetConfig(normal_cell_config=normal_cell_config)

    self._conv_sequence = darts_policies.DartsImpalaCellSequence(
        output_channels=self.output_channels, net_config=net_config)
    self._first_layer = tfkl.Conv2D(
        filters=filters[0],
        kernel_size=kernels[0],
        strides=strides[0],
        padding=padding,
        activation=None,
        dtype=dtype,
        name='%s/conv%d' % (name, 0))

    self._encoder.add(FRN())  # built-in TLU activation
    self._encoder.add(tfkl.Flatten())
    self._encoder.add(tfkl.Dense(output_dim, dtype=dtype, name='%s/fc' % name))
    self._encoder.add(tfkl.LayerNormalization(epsilon=1e-5))
    if self.output_tanh_activation:
      self._encoder.add(tfkl.Activation(tf.keras.activations.tanh))

  def call(self, inputs, step_type=None, network_state=(), training=False):
    del step_type  # unused.

    if self._uint8_input:
      inputs = tf.cast(inputs, tf.float32) / 255.00
    if self._batch_squash:
      outer_rank = nest_utils.get_outer_rank(inputs, self.input_tensor_spec)
      batch_squash = utils.BatchSquash(outer_rank)
      inputs = tf.nest.map_structure(batch_squash.flatten, inputs)
    states = inputs
    # states = self._encoder(states, training=training)
    states = self._first_layer(states, training=training)
    states = self._conv_sequence(states)
    states = self._encoder(states, training=training)
    if self._batch_squash:
      states = tf.nest.map_structure(batch_squash.unflatten, states)
    return states, network_state


@gin.configurable
class MVNormalDiagParamHead(network.Network):
  """MLP for predicting loc and scale_diag for MultivariateNormalDiag."""

  def __init__(self,
               input_tensor_spec,
               fc_layers=(128, 128),
               output_dim=50,
               scale=1.0,
               kernel_initializer='glorot_uniform',
               output_bn=True,
               batch_squash=True,
               dtype=tf.float32,
               name='MVNormalDiagParamHead'):
    """Creates an instance of `MVNormalDiagParamHead`.

    Args:
      input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
        input.
      fc_layers: A tuple. Dimensions of MLP layers.
      output_dim: An integer. Output feature dimension.
      scale: A float. If None, scale_diag is learned. Otherwise, output a fixed
        scale_diag where every dimension is scale.
      kernel_initializer: Initializer to use for the kernels of the conv and
        dense layers. If none is provided a default glorot_uniform
      output_bn: A boolean. Whether applying batch norm to the output feature.
      batch_squash: If True the outer_ranks of the observation are squashed into
        the batch dimension. This allow encoding networks to be used with
        observations with shape [BxTx...].
      dtype: The dtype to use by the convolution and fully connected layers.
      name: A string representing name of the network.
    """
    super(MVNormalDiagParamHead, self).__init__(
        input_tensor_spec=input_tensor_spec, state_spec=(), name=name)
    self.scale = scale
    self.output_dim = output_dim
    self._batch_squash = batch_squash

    self._fc_encoder = tf.keras.Sequential()
    for fc_layer_units in fc_layers:
      self._fc_encoder.add(
          tfkl.Dense(
              fc_layer_units,
              kernel_initializer=kernel_initializer,
              activation=tf.keras.activations.relu,
              dtype=dtype))
    self._fc_encoder.add(
        tfkl.Dense(
            output_dim * 2,
            kernel_initializer=tf.compat.v1.variance_scaling_initializer(1e-4),
            dtype=dtype))
    if output_bn:
      self._fc_encoder.add(tfkl.BatchNormalization())

  def call(self, inputs, step_type=None, network_state=(), training=False):
    del step_type  # unused.
    if self._batch_squash:
      outer_rank = nest_utils.get_outer_rank(inputs, self.input_tensor_spec)
      batch_squash = utils.BatchSquash(outer_rank)
      inputs = tf.nest.map_structure(batch_squash.flatten, inputs)
    states = tf.concat(inputs, axis=-1)
    states = self._fc_encoder(states, training=training)
    if self._batch_squash:
      states = tf.nest.map_structure(batch_squash.unflatten, states)
    loc = states[..., :self.output_dim]
    if self.scale is None:
      scale_diag = tf.nn.softplus(states[..., self.output_dim:])
      scale_diag *= 0.693 / tf.nn.softplus(0.)
      scale_diag += 1e-6
    else:
      scale_diag = tf.ones_like(loc) * self.scale
    return (loc, scale_diag), network_state


class FRN(tfkl.Layer):
  """Filter Response Normalization (FRN) layer with Thresholded Linear Unit.

  Filter Response Normalization Layer: Eliminating Batch Dependence in the
  Training of Deep Neural Networks
  https://arxiv.org/pdf/1911.09737.pdf
  """

  def __init__(self,
               reg_epsilon=1.0e-6,
               tau_regularizer=None,
               beta_regularizer=None,
               gamma_regularizer=None,
               **kwargs):
    """Initialize the FRN layer.

    Args:
      reg_epsilon: float, the regularization parameter preventing a division by
        zero.
      tau_regularizer: tf.keras.regularizer for tau.
      beta_regularizer: tf.keras.regularizer for beta.
      gamma_regularizer: tf.keras.regularizer for gamma.
      **kwargs: keyword arguments passed to the Keras layer base class.
    """
    self.reg_epsilon = reg_epsilon
    self.tau_regularizer = tfkr.get(tau_regularizer)
    self.beta_regularizer = tfkr.get(beta_regularizer)
    self.gamma_regularizer = tfkr.get(gamma_regularizer)
    super(FRN, self).__init__(**kwargs)

  def build(self, input_shape):
    par_shape = (1, 1, 1, input_shape[-1])  # [1,1,1,C]
    self.tau = self.add_weight(
        'tau',
        shape=par_shape,
        initializer='zeros',
        regularizer=self.tau_regularizer,
        trainable=True)
    self.beta = self.add_weight(
        'beta',
        shape=par_shape,
        initializer='zeros',
        regularizer=self.beta_regularizer,
        trainable=True)
    self.gamma = self.add_weight(
        'gamma',
        shape=par_shape,
        initializer='ones',
        regularizer=self.gamma_regularizer,
        trainable=True)

  def call(self, x):
    nu2 = tf.reduce_mean(tf.math.square(x), axis=[1, 2], keepdims=True)
    x = x * tf.math.rsqrt(nu2 + self.reg_epsilon)
    y = self.gamma * x + self.beta
    z = tf.maximum(y, self.tau)
    return z

  def get_config(self):
    config = super(FRN, self).get_config()
    config.update({
        'reg_epsilon': self.reg_epsilon,
        'tau_regularizer': tfkr.serialize(self.tau_regularizer),
        'beta_regularizer': tfkr.serialize(self.beta_regularizer),
        'gamma_regularizer': tfkr.serialize(self.gamma_regularizer),
    })
    return config
