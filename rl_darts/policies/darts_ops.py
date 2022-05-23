"""Basic DARTS Operations.

Original DARTS paper: http://arxiv.org/abs/1806.09055.
"""
import functools
from typing import Callable, Optional, Sequence

import pyglove as pg
import sonnet.v2 as snt
import tensorflow as tf

_GRAPHVIZ_URL = 'https://localhost/render_png?layout_engine=dot'


class DartsOp(snt.Module):
  """Base class for all operations used in DARTS."""

  def __init__(self, name: Optional[str] = None, **kwargs):
    """A common interface for the constructor."""
    del kwargs
    super().__init__(name)


DartsOpConstructor = Callable[..., DartsOp]


class MixedOp(DartsOp):
  """A mixture of ops.

  A `MixedOp` consists of several candidate ops. Given an input, apply all
  candidate ops to it, and return a weighted sum. The weights are provided by
  the caller (typically a `DartsCell` containing this `MixedOp`).

  We assume these candidate ops produce outputs with compatible shapes, so we
  perform the weighted sum without any checking.
  """

  def __init__(self,
               output_channels: int,
               stride: int,
               op_constructors: Sequence[DartsOpConstructor],
               name: Optional[str] = 'MixedOp'):
    super().__init__(name=name)
    self._ops = [
        op(output_channels=output_channels, stride=stride)
        for op in op_constructors
    ]

  def __call__(self, x, ops_weights, is_training):
    op_results = [op(x, is_training=is_training) for op in self._ops]
    return tf.reduce_sum(tf.stack(op_results, axis=-1) * ops_weights, axis=-1)


def generate_darts_op(
    class_name: str, simple_tf_fn: Callable[[tf.Tensor],
                                            tf.Tensor]) -> DartsOpConstructor:
  """Converts simple tf function (e.g. tf.nn.relu) into a DartsOpConstructor."""

  def init_fn(self, stride: int = 1, name: Optional[str] = None, **kwargs):
    del kwargs
    self.stride = stride
    DartsOp.__init__(self, name=name)

  def call_fn(self, x, **kwargs):  # pylint: disable=unused-argument
    del kwargs
    if self.stride > 1:
      # Normally, stride does not make sense for (non)linear ops. Here we insert
      # an average pooling layer with the specified stride. The output size
      # matches snt.Conv2D with (stride = stride, padding='SAME').
      x = tf.nn.avg_pool2d(x, ksize=3, strides=self.stride, padding='SAME')

    return simple_tf_fn(x)

  return type(class_name, (DartsOp,), {
      '__init__': init_fn,
      '__call__': call_fn
  })


class Dense(DartsOp):
  """Dense layer which can also use nonlinearities."""

  def __init__(self,
               output_channels: int,
               nonlinearity=None,
               name: Optional[str] = 'Dense',
               **kwargs):
    super().__init__(name=name)
    self._op = snt.Linear(output_size=output_channels)
    self._nonlinearity = nonlinearity

  def __call__(self, x, **kwargs):
    out = self._op(x)
    if self._nonlinearity:
      out = self._nonlinearity(out)
    return out


class Highway(DartsOp):
  """Highway bypass which is used to stabilize RNN search."""

  def __init__(self,
               output_channels: int,
               nonlinearity=None,
               name: Optional[str] = 'Highway',
               **kwargs):
    super().__init__(name=name)
    self._c_projection = snt.Linear(output_size=output_channels)
    self._h_projection = snt.Linear(output_size=output_channels)
    self._nonlinearity = nonlinearity

  def __call__(self, x, **kwargs):
    old_h = x
    c = tf.nn.sigmoid(self._c_projection(old_h))

    first_h_component = self._h_projection(old_h)
    if self._nonlinearity:
      first_h_component = self._nonlinearity(first_h_component)
    first_h_component = tf.multiply(c, first_h_component)
    second_h_component = tf.multiply(1.0 - c, old_h)

    return first_h_component + second_h_component


class MaxPool(DartsOp):
  """Max Pooling Operation.

  Stride set to 1 to preserve input shape. Use SAME padding.
  """

  def __init__(self,
               pool_size: int = 3,
               stride: int = 1,
               name: Optional[str] = 'MaxPool',
               **kwargs):
    super().__init__(name=name)
    self._op = tf.keras.layers.MaxPool2D(
        pool_size=pool_size, strides=stride, padding='SAME')

  def __call__(self, x, **kwargs):
    return self._op(x)


class AveragePool(DartsOp):
  """Average Pooling Operation.

  Stride set to 1 to preserve input shape. Use SAME padding.
  """

  def __init__(self,
               pool_size: int = 3,
               stride: int = 1,
               name: Optional[str] = 'AveragePool',
               **kwargs):
    super().__init__(name=name)
    self._op = tf.keras.layers.AveragePooling2D(
        pool_size=pool_size, strides=stride, padding='SAME')

  def __call__(self, x, **kwargs):
    return self._op(x)


class Conv(DartsOp):
  """Conv2D Wrapper.

  Stride and dilation rate are set to 1. Use SAME padding.
  """

  def __init__(self,
               output_channels: int,
               kernel_shape: int = 3,
               stride: int = 1,
               rate=1,
               nonlinearity=None,
               name: Optional[str] = 'Conv2D',
               **kwargs):
    super().__init__(name=name)
    self._op = snt.Conv2D(
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        stride=stride,
        rate=rate,
        padding='SAME')
    self._nonlinearity = nonlinearity

  def __call__(self, x, **kwargs):
    out = self._op(x)
    if self._nonlinearity:
      out = self._nonlinearity(out)
    return out


class DepthConv(DartsOp):
  """Depthwise Convolution.

  Note that this is one of 2 ops in the 'Separable' Conv Op, which is more
  frequently used in SL NAS settings.
  """

  def __init__(self,
               kernel_shape: int = 3,
               stride: int = 1,
               rate=1,
               nonlinearity=None,
               name: Optional[str] = 'DepthConv2D',
               **kwargs):
    super().__init__(name=name)

    # Setting to channel_multiplier=1 enforces same output shape as input.
    self._op = snt.DepthwiseConv2D(
        kernel_shape=kernel_shape,
        stride=1,
        channel_multiplier=1,
        rate=rate,
        padding='SAME')

    self._nonlinearity = nonlinearity

  def __call__(self, x, **kwargs):
    out = self._op(x)
    if self._nonlinearity:
      out = self._nonlinearity(out)
    return out


class DepthwiseSeparableConv(DartsOp):
  """Depthwise separable convolution.

  `DepthwiseConv2D` followed by a 1x1 conv.
  """

  def __init__(
      self,
      output_channels: int,
      kernel_shape: int = 3,
      stride: int = 1,
      rate: int = 1,
      nonlinearity=None,
      name: Optional[str] = 'DepthSeparableConv',
      **kwargs,
  ):
    super().__init__(name=name)

    # Setting to channel_multiplier=1 enforces same output shape as input.
    self._op1 = snt.DepthwiseConv2D(
        kernel_shape=kernel_shape,
        stride=stride,
        channel_multiplier=1,
        rate=rate,
        padding='SAME')

    self._op2 = snt.Conv2D(output_channels=output_channels, kernel_shape=1)
    self._nonlinearity = nonlinearity

  def __call__(self, x, **kwargs):
    out = self._op1(x)
    out = self._op2(out)
    if self._nonlinearity:
      out = self._nonlinearity(out)
    return out


class BatchNormOp(DartsOp):
  """BatchNorm wrapper."""

  def __init__(self, name: Optional[str] = None, **kwargs):
    super().__init__(name=name)
    self._op = snt.BatchNorm(create_scale=True, create_offset=True)
    del kwargs

  def __call__(self, x, is_training=True, **kwargs):
    return self._op(x, is_training)


class LayerNormOp(DartsOp):
  """LayerNorm wrapper."""

  def __init__(self, name: Optional[str] = None, **kwargs):
    super().__init__(name=name)
    self._op = snt.LayerNorm(axis=-1, create_scale=True, create_offset=True)
    del kwargs

  def __call__(self, x, **kwargs):
    return self._op(x)


OP_NAMES_TO_OP_CONSTRUCTORS = {
    'SkipConnection':
        generate_darts_op('SkipConnection', tf.identity),
    'Linear':
        functools.partial(Dense, nonlinearity=None, name='Linear'),
    'HighwayLinear':
        functools.partial(Highway, nonlinearity=None, name='HighwayLinear'),
    'Zero':
        generate_darts_op('Zero', tf.zeros_like),
    'Relu':
        generate_darts_op('Relu', tf.nn.relu),
    'DenseRelu':
        functools.partial(Dense, nonlinearity=tf.nn.relu, name='DenseRelu'),
    'HighwayRelu':
        functools.partial(
            Highway, nonlinearity=tf.nn.relu, name='HighwayLinear'),
    'Sigmoid':
        generate_darts_op('Sigmoid', tf.nn.sigmoid),
    'DenseSigmoid':
        functools.partial(
            Dense, nonlinearity=tf.nn.sigmoid, name='DenseSigmoid'),
    'HighwaySigmoid':
        functools.partial(
            Highway, nonlinearity=tf.nn.sigmoid, name='HighwaySigmoid'),
    'Tanh':
        generate_darts_op('Tanh', tf.nn.tanh),
    'DenseTanh':
        functools.partial(Dense, nonlinearity=tf.nn.tanh, name='DenseTanh'),
    'HighwayTanh':
        functools.partial(Highway, nonlinearity=tf.nn.tanh, name='HighwayTanh'),
    'Elu':
        generate_darts_op('Elu', tf.nn.elu),
    'DenseElu':
        functools.partial(Dense, nonlinearity=tf.nn.elu, name='DenseElu'),
    'HighwayElu':
        functools.partial(Highway, nonlinearity=tf.nn.elu, name='HighwayElu'),
    'Gelu':
        generate_darts_op('Gelu', tf.nn.gelu),
    'DenseGelu':
        functools.partial(Dense, nonlinearity=tf.nn.gelu, name='DenseGelu'),
    'HighwayGelu':
        functools.partial(Highway, nonlinearity=tf.nn.gelu, name='HighwayGelu'),
    'Selu':
        generate_darts_op('Selu', tf.nn.selu),
    'DenseSelu':
        functools.partial(Dense, nonlinearity=tf.nn.selu, name='DenseSelu'),
    'HighwaySelu':
        functools.partial(Highway, nonlinearity=tf.nn.selu, name='HighwaySelu'),
    'Silu':  # Also known as Swish.
        generate_darts_op('Silu', tf.nn.silu),
    'DenseSilu':
        functools.partial(Dense, nonlinearity=tf.nn.silu, name='DenseSilu'),
    'HighwaySilu':
        functools.partial(Highway, nonlinearity=tf.nn.silu, name='HighwaySilu'),
    'MaxPool2x2':
        functools.partial(MaxPool, pool_size=2),
    'MaxPool3x3':
        functools.partial(MaxPool, pool_size=3),
    'MaxPool4x4':
        functools.partial(MaxPool, pool_size=4),
    'MaxPool5x5':
        functools.partial(MaxPool, pool_size=5),
    'AveragePool2x2':
        functools.partial(AveragePool, pool_size=2),
    'AveragePool3x3':
        functools.partial(AveragePool, pool_size=3),
    'AveragePool4x4':
        functools.partial(AveragePool, pool_size=4),
    'AveragePool5x5':
        functools.partial(AveragePool, pool_size=5),
    'BatchNorm':
        BatchNormOp,
    'LayerNorm':
        LayerNormOp,
    'Conv3x3':
        functools.partial(Conv, kernel_shape=3, name='Conv3x3'),
    'Conv5x5':
        functools.partial(Conv, kernel_shape=5, name='Conv5x5'),
    'DilConv3x3':
        functools.partial(Conv, kernel_shape=3, rate=2, name='DilConv3x3'),
    'DilConv5x5':
        functools.partial(Conv, kernel_shape=5, rate=2, name='DilConv5x5'),
    'DepthConv3x3':
        functools.partial(DepthConv, kernel_shape=3, name='DepthConv3x3'),
    'DepthConv5x5':
        functools.partial(DepthConv, kernel_shape=5, name='DepthConv5x5'),
    'DepthSepConv3x3':
        functools.partial(
            DepthwiseSeparableConv, kernel_shape=3, name='DepthSepConv3x3'),
    'DepthSepConv5x5':
        functools.partial(
            DepthwiseSeparableConv, kernel_shape=5, name='DepthSepConv5x5'),
    'Conv3x3Relu':
        functools.partial(
            Conv, kernel_shape=3, nonlinearity=tf.nn.relu, name='Conv3x3Relu'),
    'Conv5x5Relu':
        functools.partial(
            Conv, kernel_shape=5, nonlinearity=tf.nn.relu, name='Conv5x5Relu'),
    'DilConv3x3Relu':
        functools.partial(
            Conv,
            kernel_shape=3,
            rate=2,
            nonlinearity=tf.nn.relu,
            name='DilConv3x3Relu'),
    'DilConv5x5Relu':
        functools.partial(
            Conv,
            kernel_shape=5,
            rate=2,
            nonlinearity=tf.nn.relu,
            name='DilConv5x5Relu'),
    'DepthConv3x3Relu':
        functools.partial(
            DepthConv,
            kernel_shape=3,
            nonlinearity=tf.nn.relu,
            name='DepthConv3x3Relu'),
    'DepthConv5x5Relu':
        functools.partial(
            DepthConv,
            kernel_shape=5,
            nonlinearity=tf.nn.relu,
            name='DepthConv5x5Relu'),
    'DepthSepConv3x3Relu':
        functools.partial(
            DepthwiseSeparableConv,
            kernel_shape=3,
            nonlinearity=tf.nn.relu,
            name='DepthSepConv3x3Relu'),
    'DepthSepConv5x5Relu':
        functools.partial(
            DepthwiseSeparableConv,
            kernel_shape=5,
            nonlinearity=tf.nn.relu,
            name='DepthSepConv5x5Relu')
}


@pg.members([
    ('op_names', pg.typing.List(pg.typing.Str()), 'Operation Names'),
])
class SearchSpace(pg.Object):
  pass


ESSENTIAL_OP_NAMES = ['SkipConnection', 'Zero']
POOLING_3x3_OP_NAMES = ['MaxPool3x3', 'AveragePool3x3']

DEFAULT_SEARCH_SPACE = SearchSpace(op_names=['Conv3x3', 'Relu', 'Tanh'] +
                                   ESSENTIAL_OP_NAMES)

# For debugging
ALL_CONV_5X5_ONLY_SEARCH_SPACE = SearchSpace(
    op_names=['Conv5x5', 'DilConv5x5', 'DepthSepConv5x5'] + ESSENTIAL_OP_NAMES)

ALL_CONV_3X3_RELU_SEARCH_SPACE = SearchSpace(
    op_names=['Conv3x3Relu', 'DilConv3x3Relu', 'DepthConv3x3Relu'] +
    ESSENTIAL_OP_NAMES)

ALL_CONV_5x5_RELU_SEARCH_SPACE = SearchSpace(
    op_names=['Conv5x5Relu', 'DilConv5x5Relu', 'DepthSepConv5x5Relu'] +
    ESSENTIAL_OP_NAMES)

# For debugging
SL_CONV_3X3_5X5_ONLY_SEARCH_SPACE = SearchSpace(
    op_names=['Conv3x3', 'DilConv3x3', 'Conv5x5', 'DilConv5x5'] +
    ESSENTIAL_OP_NAMES)

# Default SS used in DARTS.
SL_CONV_3X3_5X5_RELU_SEARCH_SPACE = SearchSpace(
    op_names=['Conv3x3Relu', 'DilConv3x3Relu', 'Conv5x5Relu', 'DilConv5x5Relu'
             ] + ESSENTIAL_OP_NAMES)
