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

"""Tests for darts_policies."""
from absl import logging
from absl.testing import parameterized

from brain_autorl.rl_darts.policies import darts_cells
from brain_autorl.rl_darts.policies import darts_policies

import numpy as np
import tensorflow as tf


class PoliciesTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    self.output_channels = 64
    self.num_nodes = 4
    self.num_inputs = 1
    self.op_names = ['Conv3x3', 'Relu', 'Tanh', 'SkipConnection', 'Zero']
    self.default_output_mode = darts_cells.CellOutputMode.CONCAT_CONV
    self.softmax_temperature = 1.0
    super().setUp()

  @parameterized.parameters(
      {'output_mode': darts_cells.CellOutputMode.LAST_NODE},
      {'output_mode': darts_cells.CellOutputMode.SUM},
      {'output_mode': darts_cells.CellOutputMode.AVERAGE},
      {'output_mode': darts_cells.CellOutputMode.WEIGHTED_SUM},
      {'output_mode': darts_cells.CellOutputMode.CONCAT_CONV},
  )
  def test_darts_impala_conv_sequence(self, output_mode):
    darts_cell_config = darts_cells.DartsCellConfig.from_create_alpha(
        op_names=self.op_names,
        num_nodes=self.num_nodes,
        output_mode=output_mode,
        trainable=False,
        softmax_temperature=self.softmax_temperature)
    net_config = darts_policies.NetConfig(
        cell_config_dict={'normal': darts_cell_config},
        config_type=darts_policies.ConfigType.DARTS)
    conv_sequence = darts_policies.DartsImpalaConvSequence(
        output_channels=self.output_channels, net_config=net_config)

    x = tf.ones((2, 16, 16, 3))
    y = conv_sequence(x)
    # Pooling operator reduces image size by half.
    self.assertEqual(y.shape, (2, 8, 8, self.output_channels))

  @parameterized.parameters(
      {'output_mode': darts_cells.CellOutputMode.LAST_NODE},
      {'output_mode': darts_cells.CellOutputMode.SUM},
      {'output_mode': darts_cells.CellOutputMode.AVERAGE},
      {'output_mode': darts_cells.CellOutputMode.WEIGHTED_SUM},
      {'output_mode': darts_cells.CellOutputMode.CONCAT_CONV},
  )
  def test_darts_impala_cnn(self, output_mode):
    darts_cell_config = darts_cells.DartsCellConfig.from_create_alpha(
        op_names=self.op_names,
        num_nodes=self.num_nodes,
        output_mode=output_mode,
        trainable=True,
        softmax_temperature=self.softmax_temperature)

    net_config = darts_policies.NetConfig(
        cell_config_dict={'normal': darts_cell_config},
        config_type=darts_policies.ConfigType.DARTS)

    darts_impala_cnn = darts_policies.DartsImpalaCNN(
        output_channels_list=[16, 32, 32], net_config=net_config)

    input_tensor = tf.random.normal(shape=(1, 84, 84, 3))
    out1 = darts_impala_cnn(input_tensor)

    darts_impala_cnn.set_train_mode(darts_policies.TrainMode.ARCH)
    impala_arch_vars = darts_impala_cnn.trainable_variables
    for var in impala_arch_vars:
      var.assign(np.random.normal(size=var.shape))

    out2 = darts_impala_cnn(input_tensor)
    self.assertNotAllEqual(out1, out2)

  def test_darts_impala_cnn_multiple_configs(self):
    darts_cell_config1 = darts_cells.DartsCellConfig.from_create_alpha(
        op_names=self.op_names,
        num_nodes=self.num_nodes,
        output_mode=darts_cells.CellOutputMode.LAST_NODE,
        trainable=True,
        softmax_temperature=self.softmax_temperature)

    darts_cell_config2 = darts_cells.DartsCellConfig.from_create_alpha(
        op_names=self.op_names,
        num_nodes=self.num_nodes,
        output_mode=darts_cells.CellOutputMode.LAST_NODE,
        trainable=True,
        softmax_temperature=self.softmax_temperature)

    net_config = darts_policies.NetConfig(
        cell_config_dict={
            'normal1': darts_cell_config1,
            'normal2': darts_cell_config2
        },
        config_type=darts_policies.ConfigType.DARTS)

    darts_impala_cnn = darts_policies.DartsImpalaCNN(
        output_channels_list=[2, 4, 8], net_config=net_config)

    darts_impala_cnn.set_train_mode(darts_policies.TrainMode.ARCH)
    self.assertLen(darts_impala_cnn.trainable_variables, self.num_nodes * 2)

  def test_darts_standard_cnn(self):
    normal_darts_cell_config = darts_cells.DartsCellConfig.from_create_alpha(
        op_names=self.op_names,
        num_nodes=self.num_nodes,
        output_mode=self.default_output_mode,
        trainable=False,
        softmax_temperature=self.softmax_temperature)

    reduction_darts_cell_config = darts_cells.DartsCellConfig.from_create_alpha(
        op_names=self.op_names,
        num_nodes=self.num_nodes,
        output_mode=self.default_output_mode,
        trainable=False,
        softmax_temperature=self.softmax_temperature,
        reduction_stride=2)

    net_config = darts_policies.NetConfig(
        cell_config_dict={
            'normal': normal_darts_cell_config,
            'reduction': reduction_darts_cell_config
        },
        config_type=darts_policies.ConfigType.DARTS)

    darts_standard_cnn = darts_policies.DartsStandardCNN(
        output_channels_list=[4, 8, 16], net_config=net_config)

    input_tensor = tf.random.normal(shape=(1, 64, 64, 3))
    # Just make sure forward pass goes through.
    darts_standard_cnn(input_tensor)

  @parameterized.parameters({'trainable_arch_vars': True},
                            {'trainable_arch_vars': False})
  def test_arch_weight_switching(self, trainable_arch_vars):
    darts_cell_config = darts_cells.DartsCellConfig.from_create_alpha(
        op_names=self.op_names,
        num_nodes=self.num_nodes,
        output_mode=self.default_output_mode,
        trainable=trainable_arch_vars,
        softmax_temperature=self.softmax_temperature)
    net_config = darts_policies.NetConfig(
        cell_config_dict={'normal': darts_cell_config},
        config_type=darts_policies.ConfigType.DARTS)

    darts_impala_cnn = darts_policies.DartsImpalaCNN(
        output_channels_list=[16, 32, 32], net_config=net_config)

    # Initialize all variables
    input_tensor = tf.random.normal(shape=(1, 84, 84, 3))
    darts_impala_cnn(input_tensor)

    darts_impala_cnn.set_train_mode(darts_policies.TrainMode.ALL)
    all_vars = darts_impala_cnn.trainable_variables
    darts_impala_cnn.set_train_mode(darts_policies.TrainMode.WEIGHT)
    weight_vars = darts_impala_cnn.trainable_variables
    darts_impala_cnn.set_train_mode(darts_policies.TrainMode.ARCH)
    arch_vars = darts_impala_cnn.trainable_variables

    logging.info('All Vars: %s', [var.name for var in all_vars])
    logging.info('Weight Vars: %s', [var.name for var in weight_vars])
    logging.info('Arch Vars: %s', [var.name for var in arch_vars])

    if trainable_arch_vars:
      self.assertEqual(
          arch_vars,
          tuple(
              darts_impala_cnn.net_config
              .get_arch_vars_and_softmax_temperatures(
                  trainable_arch_vars_only=True)[0]))
      self.assertLen(arch_vars, self.num_nodes)
      self.assertNotEqual(all_vars, weight_vars)
      expected_all_var_length = len(arch_vars) + len(weight_vars)
      self.assertLen(all_vars, expected_all_var_length)
    else:
      self.assertEmpty(arch_vars)
      self.assertEqual(all_vars, weight_vars)

  @parameterized.parameters(
      {'output_mode': darts_cells.CellOutputMode.LAST_NODE},
      {'output_mode': darts_cells.CellOutputMode.SUM},
      {'output_mode': darts_cells.CellOutputMode.AVERAGE},
      {'output_mode': darts_cells.CellOutputMode.WEIGHTED_SUM},
      {'output_mode': darts_cells.CellOutputMode.CONCAT_CONV},
  )
  def test_fixed_cell_impala_cnn(self, output_mode):
    darts_cell_config = darts_cells.DartsCellConfig.from_create_alpha(
        op_names=self.op_names,
        num_nodes=self.num_nodes,
        output_mode=output_mode,
        trainable=True,
        softmax_temperature=self.softmax_temperature)
    fixed_cell_config = darts_cell_config.to_fixed_cell_config(num_pred=2)
    net_config = darts_policies.NetConfig(
        cell_config_dict={'normal': fixed_cell_config},
        config_type=darts_policies.ConfigType.FIXED)

    model = darts_policies.DartsImpalaCNN(
        output_channels_list=[16, 16, 32], net_config=net_config)
    inputs = tf.ones([4, 32, 32, 3])
    output = model(inputs)
    self.assertEqual(tf.rank(output), 2)

  @parameterized.parameters(
      {'output_mode': darts_cells.CellOutputMode.LAST_NODE},
      {'output_mode': darts_cells.CellOutputMode.SUM},
      {'output_mode': darts_cells.CellOutputMode.AVERAGE},
      {'output_mode': darts_cells.CellOutputMode.WEIGHTED_SUM},
      {'output_mode': darts_cells.CellOutputMode.CONCAT_CONV},
  )
  def test_fixed_cell_standard_cnn(self, output_mode):
    darts_normal_cell_config = darts_cells.DartsCellConfig.from_create_alpha(
        op_names=self.op_names,
        num_nodes=self.num_nodes,
        output_mode=output_mode,
        trainable=True,
        softmax_temperature=self.softmax_temperature)

    fixed_normal_cell_config = darts_normal_cell_config.to_fixed_cell_config(
        num_pred=2)

    darts_reduction_cell_config = darts_cells.DartsCellConfig.from_create_alpha(
        op_names=self.op_names,
        num_nodes=self.num_nodes,
        output_mode=output_mode,
        trainable=True,
        softmax_temperature=self.softmax_temperature,
        reduction_stride=2)

    fixed_reduction_cell_config = darts_reduction_cell_config.to_fixed_cell_config(
        num_pred=2)

    net_config = darts_policies.NetConfig(
        cell_config_dict={
            'normal': fixed_normal_cell_config,
            'reduction': fixed_reduction_cell_config
        },
        config_type=darts_policies.ConfigType.FIXED)

    model = darts_policies.DartsStandardCNN(
        output_channels_list=[16, 16, 32], net_config=net_config)
    inputs = tf.ones([4, 32, 32, 3])
    output = model(inputs)
    self.assertEqual(tf.rank(output), 2)

  def test_darts_and_fixed_rnn_cell(self):
    batch_size = 4
    hidden_state_size = 32
    input_size = 8  # not equal to hidden_state_size

    darts_cell_config = darts_cells.DartsCellConfig.from_create_alpha(
        op_names=[
            'HighwayRelu', 'HighwayTanh', 'HighwaySigmoid', 'HighwayLinear',
            'Zero'
        ],
        num_nodes=self.num_nodes,
        output_mode=darts_cells.CellOutputMode.AVERAGE,
        trainable=True,
        softmax_temperature=self.softmax_temperature,
        use_batch_norm=True,
        num_inputs=1,
        model_type=darts_cells.ModelType.RNN)
    darts_net_config = darts_policies.NetConfig(
        cell_config_dict={
            'rnn': darts_cell_config,
        },
        config_type=darts_policies.ConfigType.DARTS)
    darts_rnn_cell = darts_policies.RNNCellNet(
        output_channels=hidden_state_size, net_config=darts_net_config)

    fixed_net_config = darts_net_config.to_fixed_net_config(num_pred=1)
    fixed_rnn_cell = darts_policies.RNNCellNet(
        output_channels=hidden_state_size, net_config=fixed_net_config)

    def assert_rnn_works(rnn_cell):
      state_tensor = rnn_cell.initial_state(batch_size=batch_size)
      for _ in range(5):
        input_tensor = tf.random.normal(shape=(batch_size, input_size))
        with tf.GradientTape() as tape:
          output_tensor, state_tensor = rnn_cell(input_tensor, state_tensor)

        self.assertEqual(output_tensor.shape, (batch_size, hidden_state_size))
        self.assertEqual(state_tensor.shape, (batch_size, hidden_state_size))

        # Check for vanishing gradients
        grads = tape.gradient(output_tensor, rnn_cell.trainable_variables)
        logging.info(grads)
        grad_norms = tf.stack([tf.norm(grad) for grad in grads])
        self.assertGreater(tf.reduce_sum(grad_norms), 0.0)

    assert_rnn_works(darts_rnn_cell)
    assert_rnn_works(fixed_rnn_cell)

  def test_hessian(self):
    """An example for computing hessian w.r.t. the arch vars."""
    darts_cell_config = darts_cells.DartsCellConfig.from_create_alpha(
        op_names=self.op_names,
        num_nodes=self.num_nodes,
        output_mode=darts_cells.CellOutputMode.CONCAT_CONV,
        trainable=True,
        softmax_temperature=self.softmax_temperature)

    net_config = darts_policies.NetConfig(
        cell_config_dict={
            'normal': darts_cell_config,
        },
        config_type=darts_policies.ConfigType.DARTS)

    darts_impala_cnn = darts_policies.DartsImpalaCNN(
        output_channels_list=[1, 1, 1], net_config=net_config)

    input_tensor = tf.random.normal(shape=(1, 8, 8, 1))
    arch_vars = [
        x for x in darts_impala_cnn.trainable_variables if 'arch_var' in x.name
    ]

    with tf.GradientTape(persistent=True) as h:
      with tf.GradientTape(persistent=True) as t:
        out1 = darts_impala_cnn(input_tensor)
        y = tf.reduce_sum(out1)
      grad1 = t.gradient(y, darts_impala_cnn.trainable_variables)
      del grad1  # We compute the usual gradient without test it.
      grad2 = t.gradient(y, arch_vars)
      # We will take the jacobian of this vector-valued function, so concat
      # along the row dimension.
      # NOTE: Another slower way to compute the jacobian is to iterate over the
      # list `grad2`, take jacobian for each of the item, then concatenate
      # results (first along columns, then along the rows).
      grad2_concat = tf.concat(grad2, axis=0)

    # 10 x 5 = 1 x 5 + 2 x 5 + 3 x 5 + 4 x 5. (4 nodes, 5 ops)
    self.assertEqual(grad2_concat.shape, [10, 5])

    hess = h.jacobian(grad2_concat, arch_vars)
    # 2nd order derivatives wrt the first (group of) arch var.
    self.assertEqual(hess[0].shape, [10, 5, 1, 5])
    # 2nd order derivatives wrt the last (group of) arch var.
    self.assertEqual(hess[3].shape, [10, 5, 4, 5])

    # Construct the Hessian matrix.
    col_blocks = []
    for x in hess:
      r, c = x.shape[0] * x.shape[1], x.shape[2] * x.shape[3]
      # The first index is for all arch vars. The second index is for one
      # (group of) arch var.
      col_blocks.append(tf.reshape(x, [r, c]))
    H = tf.concat(col_blocks, axis=1)  # pylint:disable=invalid-name
    self.assertEqual(H.shape, [50, 50])

    # Hessian should be symmetric
    self.assertAllClose(H, tf.transpose(H))

    logging.info('Largest eigenvalue: %s',
                 tf.linalg.norm(tf.linalg.eigvals(H), ord=np.inf))

    # Method 2: Lower memory consumption, but slower.
    hess = [h.jacobian(grad, arch_vars) for grad in grad2]
    row_blocks = []
    for blocks in hess:
      reshaped_blocks = []
      for b in blocks:
        r, c = b.shape[0] * b.shape[1], b.shape[2] * b.shape[3]
        reshaped_blocks.append(tf.reshape(b, [r, c]))
      row_blocks.append(tf.concat(reshaped_blocks, -1))
    H2 = tf.concat(row_blocks, axis=0)  # pylint:disable=invalid-name

    self.assertAllClose(H2, H)


if __name__ == '__main__':
  tf.test.main()
