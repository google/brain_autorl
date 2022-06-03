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

"""Tests for darts_cells."""

from absl import logging
from absl.testing import parameterized

from brain_autorl.rl_darts.policies import darts_cells
from brain_autorl.rl_darts.policies import darts_ops

import numpy as np
import pyglove as pg
import tensorflow as tf


class CellsTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    self.output_channels = 64
    self.num_nodes = 4
    self.num_inputs = 1
    self.op_names = ['Conv3x3', 'Conv5x5', 'Relu']
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
  def test_darts_cell_and_create_alpha(self, output_mode):
    darts_cell_config = darts_cells.DartsCellConfig.from_create_alpha(
        op_names=self.op_names,
        num_nodes=self.num_nodes,
        output_mode=output_mode,
        trainable=False,
        softmax_temperature=self.softmax_temperature)

    arch_vars = darts_cell_config.alpha.arch_vars
    for i in range(self.num_nodes - 1):
      self.assertEqual(arch_vars[i].shape, (i + 1, darts_cell_config.num_ops))

    cell = darts_cells.DartsCell(
        output_channels=self.output_channels, cell_config=darts_cell_config)
    inputs = [tf.random.normal(shape=(1, 84, 84, self.output_channels))]

    out1 = cell(inputs=inputs, is_training=True)

    for var in darts_cell_config.alpha.arch_vars:
      var.assign(np.random.normal(size=var.shape))

    out2 = cell(inputs=inputs, is_training=True)

    self.assertNotAllEqual(out1, out2)

  @parameterized.parameters(
      {
          'img_size': 32,
          'reduction_stride': 2
      },
      {
          'img_size': 17,
          'reduction_stride': 3
      },
  )
  def test_darts_reduction_cell(self, img_size, reduction_stride):
    darts_cell_config = darts_cells.DartsCellConfig.from_create_alpha(
        op_names=self.op_names,
        num_nodes=self.num_nodes,
        output_mode=self.default_output_mode,
        trainable=False,
        softmax_temperature=self.softmax_temperature,
        reduction_stride=reduction_stride)
    cell = darts_cells.DartsCell(
        output_channels=self.output_channels, cell_config=darts_cell_config)
    inputs = [tf.random.normal(shape=(1, img_size, img_size, 3))]
    out = cell(inputs=inputs, is_training=True)
    new_size = int(np.ceil(img_size / float(reduction_stride)))
    self.assertEqual(out.shape, (1, new_size, new_size, self.output_channels))

  @parameterized.parameters(
      {'output_mode': darts_cells.CellOutputMode.LAST_NODE},
      {'output_mode': darts_cells.CellOutputMode.SUM},
      {'output_mode': darts_cells.CellOutputMode.AVERAGE},
      {'output_mode': darts_cells.CellOutputMode.WEIGHTED_SUM},
      {'output_mode': darts_cells.CellOutputMode.CONCAT_CONV},
  )
  def test_darts_cell_save_and_visualize(self, output_mode):
    arch_vars = []
    arch_vars.append(tf.Variable([[1.0, 1.0, 1.0]]))
    arch_vars.append(tf.Variable([[1.0, 0.0, 0.0], [0.1, -0.1, 0.0]]))
    if output_mode == darts_cells.CellOutputMode.WEIGHTED_SUM:
      arch_vars.append(tf.Variable([[1.0, 0.0]]))

    alpha = darts_cells.Alpha(
        arch_vars=arch_vars,
        softmax_temperature=tf.Variable(1.0, trainable=False))
    logging.info(alpha.variables)
    darts_cell_config = darts_cells.DartsCellConfig(
        search_space=darts_ops.SearchSpace(op_names=['op1', 'op2', 'op3']),
        output_mode=output_mode)
    darts_cell_config.alpha = alpha

    dir_path = self.get_temp_dir()
    darts_cell_config.save(dir_path)
    darts_cell_config_2 = darts_cells.DartsCellConfig.load(dir_path)

    # This only checks if PyGlove symbolics are equal. Arch vars are checked
    # below.
    self.assertEqual(darts_cell_config, darts_cell_config_2)

    self.assertAllEqual(darts_cell_config.alpha.softmax_temperature,
                        darts_cell_config_2.alpha.softmax_temperature)

    for i, arch_var in enumerate(darts_cell_config.alpha.arch_vars):
      self.assertAllEqual(arch_var, darts_cell_config_2.alpha.arch_vars[i])

    self.assertAllClose(darts_cell_config_2.alpha.arch_probs()[0],
                        tf.constant([[1. / 3, 1. / 3, 1. / 3]]), 1e-6)
    logging.info('graphviz url: %s', darts_cell_config.visualize_graph_url())

  def test_arch_var_entropy(self):
    softmax_temperature = 1.0
    arch_var = tf.constant([[1.0, 0.0], [0.1, -0.1]])
    mean_entropy = darts_cells.arch_var_entropy(arch_var, softmax_temperature)

    uniform_arch_var = tf.constant([[0.0, 0.0], [1.0, 1.0]])
    uniform_mean_entropy = darts_cells.arch_var_entropy(uniform_arch_var,
                                                        softmax_temperature)

    logging.info('Mean Entropy: %f', mean_entropy)
    logging.info('Max Uniform Entropy: %f', uniform_mean_entropy)

    self.assertLessEqual(mean_entropy, uniform_mean_entropy)

  def test_fixed_cell_save_and_visualize(self):
    arch_vars = []
    arch_vars.append(tf.constant([[1., 0., 0.]]))
    arch_vars.append(tf.constant([[1., 0., 0.], [0., 0., 1.]]))
    arch_vars.append(tf.constant([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]))

    alpha = darts_cells.Alpha(
        arch_vars=arch_vars,
        softmax_temperature=tf.Variable(1.0, trainable=False))
    darts_cell_config = darts_cells.DartsCellConfig(
        search_space=darts_ops.SearchSpace(op_names=['op1', 'op2', 'op3']),
        output_mode=self.default_output_mode)
    darts_cell_config.alpha = alpha
    fixed_cell_config = darts_cell_config.to_fixed_cell_config(num_pred=2)

    filepath = self.create_tempfile()
    fixed_cell_config.save(filepath)
    fixed_cell_config_2 = pg.load(filepath)

    self.assertEqual(fixed_cell_config, fixed_cell_config_2)

    logging.info('graphviz url: %s', fixed_cell_config.visualize_graph_url())

  def test_fixed_cell_weighted_sum(self):
    arch_vars = []
    arch_vars.append(tf.constant([[1., 0., 0.]]))
    arch_vars.append(tf.constant([[1., 0., 0.], [0., 0., 1.]]))
    arch_vars.append(tf.constant([[0., 0.]]))

    alpha = darts_cells.Alpha(
        arch_vars=arch_vars,
        softmax_temperature=tf.Variable(1.0, trainable=False))
    darts_cell_config = darts_cells.DartsCellConfig(
        search_space=darts_ops.SearchSpace(op_names=['op1', 'op2', 'op3']),
        output_mode=darts_cells.CellOutputMode.WEIGHTED_SUM)
    darts_cell_config.alpha = alpha

    fixed_cell_config = darts_cell_config.to_fixed_cell_config(num_pred=2)
    self.assertAllClose(fixed_cell_config.output_weights, [0.5, 0.5])

  @parameterized.parameters(
      {'output_mode': darts_cells.CellOutputMode.LAST_NODE},
      {'output_mode': darts_cells.CellOutputMode.SUM},
      {'output_mode': darts_cells.CellOutputMode.AVERAGE},
      {'output_mode': darts_cells.CellOutputMode.WEIGHTED_SUM},
      {'output_mode': darts_cells.CellOutputMode.CONCAT_CONV},
  )
  def test_fixed_cell(self, output_mode):
    darts_cell_config = darts_cells.DartsCellConfig.from_create_alpha(
        op_names=self.op_names,
        num_nodes=self.num_nodes,
        output_mode=output_mode,
        trainable=False,
        softmax_temperature=self.softmax_temperature)
    fixed_cell_config = darts_cell_config.to_fixed_cell_config(num_pred=2)

    fixed_cell = darts_cells.FixedCell(
        output_channels=self.output_channels, cell_config=fixed_cell_config)
    inputs = [tf.random.normal(shape=(4, 32, 32, 16))]
    out = fixed_cell(inputs)
    self.assertEqual(out.shape, [4, 32, 32, self.output_channels])

  @parameterized.parameters(
      {
          'img_size': 32,
          'reduction_stride': 2
      },
      {
          'img_size': 17,
          'reduction_stride': 3
      },
  )
  def test_fixed_reduction_cell(self, img_size, reduction_stride):
    darts_cell_config = darts_cells.DartsCellConfig.from_create_alpha(
        op_names=self.op_names,
        num_nodes=self.num_nodes,
        output_mode=self.default_output_mode,
        trainable=False,
        softmax_temperature=self.softmax_temperature,
        reduction_stride=reduction_stride)
    fixed_cell_config = darts_cell_config.to_fixed_cell_config(num_pred=2)

    cell = darts_cells.FixedCell(
        output_channels=self.output_channels, cell_config=fixed_cell_config)
    inputs = [tf.random.normal(shape=(4, img_size, img_size, 3))]
    out = cell(inputs)
    new_size = int(np.ceil(img_size / float(reduction_stride)))
    self.assertEqual(out.shape, [4, new_size, new_size, self.output_channels])


if __name__ == '__main__':
  tf.test.main()
