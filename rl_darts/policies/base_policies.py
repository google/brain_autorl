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

# pylint: disable=missing-class-docstring,missing-function-docstring, dangerous-default-value
"""Basic vanilla neural networks."""
from typing import Sequence, Optional
import sonnet.v2 as snt
import tensorflow as tf


class ImpalaConvLayer(snt.Module):

  def __init__(self,
               depth: int,
               dropout_rate: float = 0.0,
               use_batch_norm: bool = False,
               name: Optional[str] = None,
               **kwargs):
    del kwargs
    super(ImpalaConvLayer, self).__init__(name=name)
    self.conv = snt.Conv2D(depth, 3, padding='SAME')
    self.bn = snt.BatchNorm(create_scale=True, create_offset=True)
    self.dropout_rate = dropout_rate
    self.use_batch_norm = use_batch_norm

  def __call__(self, inputs, is_training=True, **kwargs):
    del kwargs
    out = self.conv(inputs)
    if is_training:
      out = tf.nn.dropout(out, rate=self.dropout_rate)
    if self.use_batch_norm:
      out = self.bn(out, is_training=is_training)
    return out


class ImpalaResidualBlock(snt.Module):

  def __init__(self,
               depth: int,
               conv_layer=ImpalaConvLayer,
               name: Optional[str] = None,
               **kwargs):
    super(ImpalaResidualBlock, self).__init__(name=name)
    self.conv1 = conv_layer(depth=depth, name='c1', **kwargs)
    self.conv2 = conv_layer(depth=depth, name='c2', **kwargs)

  def __call__(self, inputs, is_training=True, **kwargs):
    out = tf.nn.relu(inputs)
    out = self.conv1(out, is_training=is_training, **kwargs)
    out = tf.nn.relu(out)
    out = self.conv2(out, is_training=is_training, **kwargs)
    return out + inputs


class ImpalaConvSequence(snt.Module):

  def __init__(self,
               depth: int,
               conv_layer=ImpalaConvLayer,
               residual_block=ImpalaResidualBlock,
               name: Optional[str] = None,
               **kwargs):
    super(ImpalaConvSequence, self).__init__(name=name)
    self.conv = conv_layer(depth=depth, name='c', **kwargs)
    self.residual1 = residual_block(
        depth=depth, conv_layer=conv_layer, name='r1', **kwargs)
    self.residual2 = residual_block(
        depth=depth, conv_layer=conv_layer, name='r2', **kwargs)

  def __call__(self, inputs, is_training=True, **kwargs):
    out = self.conv(inputs, is_training=is_training, **kwargs)
    out = tf.nn.max_pool2d(out, ksize=3, strides=2, padding='SAME')
    out = self.residual1(out, is_training=is_training, **kwargs)
    out = self.residual2(out, is_training=is_training, **kwargs)
    return out


class ImpalaCNN(snt.Module):

  def __init__(self,
               impala_sequence=ImpalaConvSequence,
               depths: Sequence[int] = [16, 32, 32],
               mlp_size: int = 256,
               name: Optional[str] = None,
               **kwargs):
    super(ImpalaCNN, self).__init__(name=name)

    temp_list = []
    for d in depths:
      temp_list.append(
          impala_sequence(depth=d, name='impala_conv_seq' + str(d), **kwargs))

    self.conv_section = snt.Sequential(temp_list)
    self.linear = snt.Linear(mlp_size)

  def __call__(self, inputs, is_training=True, **kwargs):
    out = self.conv_section(inputs, is_training=is_training, **kwargs)
    # Since inputs must be images, inner_rank is always 3.
    # This setting for Flatten() is useful if e.g. tensor has an extra time dim.
    outer_rank = len(out.shape) - 3
    out = snt.Flatten(preserve_dims=outer_rank)(out)
    out = tf.nn.relu(out)
    out = self.linear(out)
    out = tf.nn.relu(out)
    return out


class CustomMLP(snt.nets.MLP):
  """More flexible MLP for AutoRL specific uses.

  Original MLP has too many restrictions.
  """

  def __init__(self, input_inner_rank: int, **kwargs):
    super().__init__(**kwargs)
    # Rank of tensor that's not batch or time dimensions.
    # Ex: If dealing with [H, W, C] image inputs, set to 3.
    self._input_inner_rank = input_inner_rank

  def __call__(self, inputs: tf.Tensor, is_training=None) -> tf.Tensor:
    """Connects the module to some inputs.

    Args:
      inputs: A Tensor of any shape `[batch_size, ...]`.
      is_training: A bool indicating if we are currently training. Defaults to
        `None`. Required if using dropout.

    Returns:
      output: The output of the model of size `[batch_size, output_size]`.
    """
    outer_rank = len(inputs.shape) - self._input_inner_rank
    inputs = snt.Flatten(preserve_dims=outer_rank)(inputs)
    num_layers = len(self._layers)

    for i, layer in enumerate(self._layers):
      inputs = layer(inputs)
      if i < (num_layers - 1) or self._activate_final:
        inputs = self._activation(inputs)

    return inputs


def make_impala_cnn_network(conv_layer=ImpalaConvLayer,
                            depths: Sequence[int] = [16, 32, 32],
                            mlp_size: int = 256,
                            use_batch_norm: bool = False,
                            dropout_rate: float = 0.0):
  return ImpalaCNN(
      depths=depths,
      mlp_size=mlp_size,
      name='impala',
      use_batch_norm=use_batch_norm,
      dropout_rate=dropout_rate,
      conv_layer=conv_layer)
