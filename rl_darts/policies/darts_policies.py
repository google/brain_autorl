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

"""RL Policies constructed via DARTS."""
import abc
import enum
from typing import List, Optional, Tuple, Dict

from brain_autorl.rl_darts.policies import darts_cells
from brain_autorl.rl_darts.policies import darts_ops

import sonnet.v2 as snt
import tensorflow as tf


@enum.unique
class TrainMode(enum.Enum):
  """Sets what the trainable variables are in a DartsNet.

  Useful for bilevel/alternating optimization between architecture variables
  and regular weights.
  """
  ALL = 'all'
  ARCH = 'arch'
  WEIGHT = 'weight'


@enum.unique
class ConfigType(str, enum.Enum):
  """DARTS or Fixed cell config."""
  DARTS = 'darts'
  FIXED = 'fixed'


class NetConfig():
  """Container for Regular DARTS network configs.

  Mainly done via mapping an identifier string to cell config. Downstream
  networks will recognize those identifier strings.
  """

  def __init__(self,
               cell_config_dict: Dict[str, darts_cells.CellConfig],
               config_type: ConfigType = ConfigType.DARTS):
    self._cell_config_dict = cell_config_dict
    self._config_type = config_type

  @property
  def config_type(self) -> ConfigType:
    return self._config_type

  @property
  def cell_config_dict(self) -> Dict[str, darts_cells.CellConfig]:
    return self._cell_config_dict

  def to_fixed_net_config(self, num_pred: int = 2) -> 'NetConfig':
    """Convenience function for fixed cell config conversions."""
    if self._config_type == ConfigType.FIXED:
      raise ValueError('Already a fixed cell config.')

    cell_config_dict = {}
    for cell_config_name, cell_config in self._cell_config_dict.items():
      cell_config_dict[cell_config_name] = cell_config.to_fixed_cell_config(
          num_pred=num_pred)

    return NetConfig(
        cell_config_dict=cell_config_dict, config_type=ConfigType.FIXED)

  def get_arch_vars_and_softmax_temperatures(
      self, trainable_arch_vars_only: bool
  ) -> Tuple[List[tf.Variable], List[tf.Variable]]:
    """Get all arch_vars and softmax_temperatures from all configs.

    Can filter arch_vars for trainability. Needed for variable tracking during
    the __init__ call.

    Args:
      trainable_arch_vars_only: For arch_vars, only return trainable ones.

    Returns:
      architecture variables and softmax_temperature variables.
    """
    all_arch_variables = []
    all_softmax_temperatures = []
    for config in self._cell_config_dict.values():
      if isinstance(config, darts_cells.DartsCellConfig):
        if trainable_arch_vars_only:
          for arch_var in config.alpha.arch_vars:
            if arch_var.trainable:
              all_arch_variables.append(arch_var)
        else:
          all_arch_variables.extend(config.alpha.arch_vars)
        all_softmax_temperatures.append(config.alpha.softmax_temperature)
    return all_arch_variables, all_softmax_temperatures

  def total_mean_entropy(self) -> darts_cells.FloatLike:
    all_entropies = []
    for config in self._cell_config_dict.values():
      if isinstance(config, darts_cells.DartsCellConfig):
        all_entropies.append(config.alpha.total_mean_entropy())

    return tf.reduce_mean(tf.stack(all_entropies))


class DartsNet(snt.Module, abc.ABC):
  """Base class for all DARTS networks, allowing both supernet (for search) + discrete (for eval) creations."""

  def __init__(self, net_config: NetConfig, name: str):
    super().__init__(name)
    self._net_config = net_config
    if self._net_config.config_type == ConfigType.DARTS:
      self.init_from_darts_net_config(self._net_config)
    elif self._net_config.config_type == ConfigType.FIXED:
      self.init_from_fixed_net_config(self._net_config)
    else:
      raise ValueError('Unknown CellConfig Type.')

    self._trainable_arch_vars, self._softmax_temperatures = self._net_config.get_arch_vars_and_softmax_temperatures(
        trainable_arch_vars_only=True)  # For variable tracking.

    self.set_train_mode(TrainMode.ALL)  # Default.

  @property
  def net_config(self) -> NetConfig:
    return self._net_config

  @abc.abstractmethod
  def init_from_darts_net_config(self, net_config: NetConfig):
    pass

  @abc.abstractmethod
  def init_from_fixed_net_config(self, net_config: NetConfig):
    pass

  def set_train_mode(self, mode: TrainMode):
    self._mode = mode

  @property
  def trainable_variables(self):
    # fget is used to force out all trainable_variables in a @property.
    all_variables = snt.Module.trainable_variables.fget(self)

    if self._mode == TrainMode.ALL:
      return all_variables

    elif self._mode == TrainMode.ARCH:
      return tuple(self._trainable_arch_vars)

    elif self._mode == TrainMode.WEIGHT:
      all_arch_variable_names = [v.name for v in self._trainable_arch_vars]
      weight_variables = []
      for variable in all_variables:
        if variable.name not in all_arch_variable_names:
          weight_variables.append(variable)
      return tuple(weight_variables)

    else:
      raise ValueError('Wrong mode was used.')


class RNNCellNet(snt.RNNCore, DartsNet):
  """Wraps a regular single cell as a DartsNet."""

  def __init__(self,
               output_channels: int,
               net_config: NetConfig,
               name: Optional[str] = 'RNNCellNet'):
    assert 'rnn' in net_config.cell_config_dict

    self._output_channels = output_channels
    self._hidden_size = output_channels  # For KerasRNN conversions.
    DartsNet.__init__(self, net_config, name=name)

    self._x_to_c_proj = snt.Linear(output_size=output_channels)
    self._h_to_c_proj = snt.Linear(output_size=output_channels)
    self._x_to_new_h_proj = snt.Linear(output_size=output_channels)
    self._h_to_new_h_proj = snt.Linear(output_size=output_channels)

  def __call__(self, inputs, prev_state, is_training=True):
    # Highway bypass for input and rnn_state.
    x = inputs
    old_h = prev_state
    c = tf.nn.sigmoid(self._x_to_c_proj(x) + self._h_to_c_proj(old_h))

    # tanh was fixed in DARTS paper, but searchable in ENAS.
    first_new_h_component = tf.nn.tanh(
        self._x_to_new_h_proj(x) + self._h_to_new_h_proj(old_h))
    first_new_h_component = tf.multiply(c, first_new_h_component)
    second_new_h_component = tf.multiply(1.0 - c, old_h)
    new_h = first_new_h_component + second_new_h_component

    new_state = self._cell([new_h], is_training)
    outputs = new_state
    # Next state of RNN is same as output.
    return outputs, new_state

  def initial_state(self, batch_size, **kwargs):
    return tf.zeros(shape=[batch_size, self._output_channels])

  def init_from_darts_net_config(self, net_config: NetConfig):
    self._cell = darts_cells.DartsCell(
        output_channels=self._output_channels,
        cell_config=net_config.cell_config_dict['rnn'])

  def init_from_fixed_net_config(self, net_config: NetConfig):
    self._cell = darts_cells.FixedCell(
        output_channels=self._output_channels,
        cell_config=net_config.cell_config_dict['rnn'])


class DartsImpalaConvSequence(DartsNet):
  """A DARTS version of ImpalaConvSequence."""

  def __init__(self,
               output_channels: int,
               net_config: NetConfig,
               name: Optional[str] = 'DartsImpalaConvSequence'):
    if len(net_config.cell_config_dict) == 1:
      assert 'normal' in net_config.cell_config_dict
      self._normal_key1, self._normal_key2 = 'normal', 'normal'
    elif len(net_config.cell_config_dict) == 2:
      assert 'normal1' in net_config.cell_config_dict
      assert 'normal2' in net_config.cell_config_dict
      self._normal_key1, self._normal_key2 = 'normal1', 'normal2'
    else:
      raise ValueError('net_config.cell_config_dict has unsupported length: '
                       f'{len(net_config.cell_config_dict)}')

    self._output_channels = output_channels
    super().__init__(net_config, name=name)
    self._conv = darts_ops.Conv(output_channels=self._output_channels)
    self._bn = snt.BatchNorm(create_scale=True, create_offset=True)

  def init_from_darts_net_config(self, net_config: NetConfig):
    self._cell_1 = darts_cells.DartsCell(
        output_channels=self._output_channels,
        cell_config=net_config.cell_config_dict[self._normal_key1])
    self._cell_2 = darts_cells.DartsCell(
        output_channels=self._output_channels,
        cell_config=net_config.cell_config_dict[self._normal_key2])

  def init_from_fixed_net_config(self, net_config: NetConfig):
    self._cell_1 = darts_cells.FixedCell(
        output_channels=self._output_channels,
        cell_config=net_config.cell_config_dict[self._normal_key1])
    self._cell_2 = darts_cells.FixedCell(
        output_channels=self._output_channels,
        cell_config=net_config.cell_config_dict[self._normal_key2])

  def __call__(self, x, is_training=True):
    x = self._conv(x)
    # x = self._bn(x, is_training)  # disabling for now.
    x = tf.nn.max_pool2d(x, ksize=3, strides=2, padding='SAME')
    x = self._cell_1([x], is_training)
    x = self._cell_2([x], is_training)
    return x


class DartsImpalaCNN(DartsNet):
  """A DARTS version of ImpalaCNN."""

  def __init__(
      self,
      output_channels_list: List[int],
      net_config: NetConfig,
      name: Optional[str] = 'DartsImpalaCNN',
  ):
    self._conv_sequences = []
    self._output_channels_list = output_channels_list
    super().__init__(net_config, name=name)
    self._linear = snt.Linear(256)

  def init_from_darts_net_config(self, net_config: NetConfig):
    for output_channels in self._output_channels_list:
      temp_conv_sequence = DartsImpalaConvSequence(output_channels, net_config)
      self._conv_sequences.append(temp_conv_sequence)

  def init_from_fixed_net_config(self, net_config: NetConfig):
    for output_channels in self._output_channels_list:
      temp_conv_sequence = DartsImpalaConvSequence(output_channels, net_config)
      self._conv_sequences.append(temp_conv_sequence)

  def __call__(self, x, is_training=True):
    for conv_sequence in self._conv_sequences:
      x = conv_sequence(x, is_training)
    # Since inputs must be images, inner_rank is always 3.
    # This setting for Flatten() is useful if e.g. tensor has an extra time dim.
    outer_rank = len(x.shape) - 3
    x = snt.Flatten(preserve_dims=outer_rank)(x)
    x = tf.nn.relu(x)
    x = self._linear(x)
    x = tf.nn.relu(x)
    return x


class DartsStandardCNN(DartsNet):
  """A standard Darts CNN Supernet that has reduction cells, based off of IMPALA-CNN."""

  def __init__(
      self,
      output_channels_list: List[int],
      net_config: NetConfig,
      use_initial_conv: bool = True,
      name: Optional[str] = 'DartsStandardCNN',
  ):
    assert 'normal' in net_config.cell_config_dict
    assert 'reduction' in net_config.cell_config_dict

    self._cells = []
    self._output_channels_list = output_channels_list
    super().__init__(net_config, name=name)

    # B/c reduction cell is first, this avoids reducing image size too early.
    # We still use reduction cell first to avoid memory issues.
    self.use_initial_conv = use_initial_conv
    if self.use_initial_conv:
      self.first_conv = snt.Conv2D(
          output_channels=output_channels_list[0], kernel_shape=3)

    self.linear = snt.Linear(256)

  def init_from_darts_net_config(self, net_config: NetConfig):
    for output_channels in self._output_channels_list:
      self._cells.append(
          darts_cells.DartsCell(
              output_channels=output_channels,
              cell_config=net_config.cell_config_dict['reduction']))
      self._cells.append(
          darts_cells.DartsCell(
              output_channels=output_channels,
              cell_config=net_config.cell_config_dict['normal']))
      self._cells.append(
          darts_cells.DartsCell(
              output_channels=output_channels,
              cell_config=net_config.cell_config_dict['normal']))

  def init_from_fixed_net_config(self, net_config: NetConfig):
    for output_channels in self._output_channels_list:
      self._cells.append(
          darts_cells.FixedCell(
              output_channels=output_channels,
              cell_config=net_config.cell_config_dict['reduction']))
      self._cells.append(
          darts_cells.FixedCell(
              output_channels=output_channels,
              cell_config=net_config.cell_config_dict['normal']))
      self._cells.append(
          darts_cells.FixedCell(
              output_channels=output_channels,
              cell_config=net_config.cell_config_dict['normal']))

  def __call__(self, x, is_training=True, darts_output=False):
    if self.use_initial_conv:
      x = self.first_conv(x)

    for cell in self._cells:
      x = cell([x], is_training)

    if darts_output:
      # Sometimes we just want these raw feature maps.
      return x
    # Since inputs must be images, inner_rank is always 3.
    # This setting for Flatten() is useful if e.g. tensor has an extra time dim.
    outer_rank = len(x.shape) - 3
    x = snt.Flatten(outer_rank)(x)
    x = tf.nn.relu(x)
    x = self.linear(x)
    x = tf.nn.relu(x)
    return x
