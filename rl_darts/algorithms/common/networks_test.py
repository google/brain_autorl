"""Tests for networks."""
from absl.testing import parameterized

from brain_autorl.rl_darts.algorithms.common import networks
from brain_autorl.rl_darts.policies import base_policies
from brain_autorl.rl_darts.policies import darts_cells
from brain_autorl.rl_darts.policies import darts_policies

import sonnet.v2 as snt
import tensorflow as tf
from tf_agents.keras_layers import dynamic_unroll_layer
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts


class NetworksTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    self.default_batch_size = 3
    self.default_rnn_units = 8
    self.default_mlp_size = 16
    self.default_num_actions = 5

    self.darts_cell_config = darts_cells.DartsCellConfig.from_create_alpha(
        op_names=[
            'HighwayRelu', 'HighwayTanh', 'HighwaySigmoid', 'HighwayLinear',
            'Zero'
        ],
        num_nodes=5,
        output_mode=darts_cells.CellOutputMode.AVERAGE,
        trainable=True,
        softmax_temperature=5.0,
        use_batch_norm=False,
        num_inputs=1,
        model_type=darts_cells.ModelType.RNN)
    self.darts_net_config = darts_policies.NetConfig(
        {'rnn': self.darts_cell_config})
    super().setUp()

  @parameterized.parameters({'use_rnn': False}, {'use_rnn': True})
  def test_ppo_networks(self, use_rnn):
    feature_network = base_policies.make_impala_cnn_network(
        mlp_size=self.default_mlp_size)
    if use_rnn:
      rnn_cell = tf.keras.layers.SimpleRNNCell(units=self.default_rnn_units)
    else:
      rnn_cell = None

    observation_spec = tensor_spec.BoundedTensorSpec((64, 64, 3), tf.float32, 0,
                                                     1)
    action_spec = tensor_spec.BoundedTensorSpec(
        (self.default_num_actions,), tf.int32, 0, self.default_num_actions)

    time_step_spec = ts.time_step_spec(observation_spec)
    time_step = tensor_spec.sample_spec_nest(
        time_step_spec, outer_dims=(self.default_batch_size,))

    encoder = networks.CustomEncodingNetwork(
        input_tensor_spec=observation_spec,
        feature_network=feature_network,
        rnn_cell=rnn_cell)
    output, rnn_state = encoder(
        time_step.observation,
        time_step.step_type,
        encoder.get_initial_state(batch_size=self.default_batch_size),
        training=False)
    if use_rnn:
      self.assertEqual(output.shape,
                       (self.default_batch_size, self.default_rnn_units))
      self.assertEqual(rnn_state.shape,
                       (self.default_batch_size, self.default_rnn_units))
    else:
      self.assertEqual(output.shape,
                       (self.default_batch_size, self.default_mlp_size))

    actor_net = networks.CustomActorDistributionNetwork(action_spec, encoder)
    action_distribution, rnn_state = actor_net(
        time_step.observation, time_step.step_type,
        actor_net.get_initial_state(batch_size=self.default_batch_size))

    self.assertEqual(action_distribution.mode().shape,
                     (self.default_batch_size, self.default_num_actions))
    if use_rnn:
      self.assertEqual(rnn_state.shape,
                       (self.default_batch_size, self.default_rnn_units))

    value_net = networks.CustomValueNetwork(encoder)
    value, rnn_state = value_net(
        time_step.observation,
        step_type=time_step.step_type,
        network_state=value_net.get_initial_state(
            batch_size=self.default_batch_size))
    self.assertEqual(value.shape, (self.default_batch_size,))
    if use_rnn:
      self.assertEqual(rnn_state.shape,
                       (self.default_batch_size, self.default_rnn_units))

  @parameterized.parameters({'rnn_constructor': snt.VanillaRNN},
                            {'rnn_constructor': darts_policies.RNNCellNet})
  def test_sonnet_rnn(self, rnn_constructor):
    max_time = 7

    if rnn_constructor == darts_policies.RNNCellNet:
      snt_rnn_cell = rnn_constructor(
          self.default_rnn_units, net_config=self.darts_net_config)
    else:
      snt_rnn_cell = rnn_constructor(self.default_rnn_units)
    keras_rnn_cell = networks.SonnetRNNCelltoKerasRNNCell(
        sonnet_rnn=snt_rnn_cell)
    unrolled_rnn = dynamic_unroll_layer.DynamicUnroll(keras_rnn_cell)
    inputs = tf.random.uniform((self.default_batch_size, max_time, 2),
                               dtype=tf.float32)

    # Check gradients pass through.
    @tf.function
    def unroll(rnn, inp):
      return rnn(inp)

    with tf.GradientTape() as tape:
      outputs, final_state = unroll(unrolled_rnn, inputs)

    self.assertEqual(
        outputs.shape,
        (self.default_batch_size, max_time, self.default_rnn_units))
    self.assertEqual(final_state[0].shape,
                     (self.default_batch_size, self.default_rnn_units))

    grads = tape.gradient(outputs, keras_rnn_cell.trainable_weights)
    for i, grad in enumerate(grads):
      self.assertEqual(grad.shape, keras_rnn_cell.trainable_weights[i].shape)

    # Check for vanishing gradients.
    grad_norms = tf.stack([tf.norm(grad) for grad in grads])
    self.assertGreater(tf.reduce_sum(grad_norms), 0.0)


if __name__ == '__main__':
  tf.test.main()
