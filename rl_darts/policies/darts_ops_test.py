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

"""Tests for darts_ops."""

from absl.testing import parameterized

from brain_autorl.rl_darts.policies import darts_ops

import tensorflow as tf


class OperationsTest(tf.test.TestCase, parameterized.TestCase):

  def test_mixed_op(self):
    op_constructors = [
        darts_ops.generate_darts_op('Zero', tf.zeros_like),
        darts_ops.generate_darts_op('Id', tf.identity),
    ]
    # output_channels arg is ignored.
    mixed_op = darts_ops.MixedOp(
        output_channels=0, stride=1, op_constructors=op_constructors)
    x = tf.ones((1, 3, 4, 2))
    y = mixed_op(x, ops_weights=[0.5, 0.5], is_training=True)
    self.assertAllClose(y, 0.5 * x)

  @parameterized.parameters({'reduction_stride': 1}, {'reduction_stride': 2},
                            {'reduction_stride': 3}, {'reduction_stride': 4},
                            {'reduction_stride': 5})
  def test_generate_nonlinearity_function(self, reduction_stride):
    relu_darts_op = darts_ops.generate_darts_op('Relu', tf.nn.relu)
    inp = tf.random.normal(shape=(1, 84, 84, 3))

    darts_op = relu_darts_op(stride=reduction_stride)
    darts_op_output = darts_op(inp)

    if reduction_stride == 1:
      expected_output = tf.nn.relu(inp)
      self.assertAllEqual(darts_op_output, expected_output)

    # Verifies striding for nonlinearity has same behavior as regular Conv.
    conv_op = darts_ops.OP_NAMES_TO_OP_CONSTRUCTORS['Conv3x3'](
        output_channels=3, stride=reduction_stride)
    conv_op_output = conv_op(inp)
    self.assertEqual(darts_op_output.shape, conv_op_output.shape)

  @parameterized.parameters({'reduction_stride': 1}, {'reduction_stride': 2},
                            {'reduction_stride': 3}, {'reduction_stride': 4},
                            {'reduction_stride': 5})
  def test_conv_and_pool_output_shapes(self, reduction_stride):
    output_channels = 16
    conv_and_pool_op_names = [
        'Conv3x3', 'Conv5x5', 'MaxPool3x3', 'AveragePool5x5'
    ]

    ops = [
        darts_ops.OP_NAMES_TO_OP_CONSTRUCTORS[op_name](
            output_channels=output_channels, stride=reduction_stride)
        for op_name in conv_and_pool_op_names
    ]

    inp = tf.random.normal(shape=(1, 84, 84, output_channels))
    outputs = [op(inp) for op in ops]

    for i in range(len(outputs)):
      for j in range(i + 1, len(outputs)):
        self.assertEqual(outputs[i].shape, outputs[j].shape)


if __name__ == '__main__':
  tf.test.main()
