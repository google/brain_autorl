"""Basic building blocks (ex: cells) for DARTS architectures."""

import abc
import enum
import os
from typing import List, Optional, Union, TypeVar
import urllib

from brain_autorl.rl_darts.policies import darts_ops

import numpy as np
import pydot
import pyglove as pg
import sonnet.v2 as snt
import tensorflow as tf

FloatLike = Union[tf.Tensor, float, np.ndarray]

_GRAPHVIZ_URL = 'https://localhost/render_png?layout_engine=dot'


@enum.unique
class CellOutputMode(str, enum.Enum):
  """Represents how output is generated from intermediate nodes."""

  # Basic sum of node outputs.
  SUM = 'sum'

  # Uniform average of node inputs.
  AVERAGE = 'average'

  # Weighted sum of intermediate nodes via the final arch_var.
  WEIGHTED_SUM = 'weighted_sum'

  # Concats intermediate nodes depth-wise; followed by 1x1 conv.
  CONCAT_CONV = 'concat_conv'

  # Uses last node as output.
  LAST_NODE = 'last_node'

  @staticmethod
  def list():
    return list(map(lambda c: c.value, CellOutputMode))


@enum.unique
class ModelType(str, enum.Enum):
  """CNN or RNNs."""
  # For inputs of shape [B, H, W, C].
  CNN = 'cnn'

  # For inputs of shape [B, D].
  RNN = 'rnn'

  @staticmethod
  def list():
    return list(map(lambda c: c.value, ModelType))


def arch_var_entropy(arch_var: tf.Variable,
                     softmax_temperature: float) -> FloatLike:
  """Gives the entropy of a single MixedOp's arch_var."""
  prob = tf.nn.softmax(softmax_temperature * arch_var, axis=-1)
  neg_p_log_p_vals = -1.0 * tf.reduce_sum(prob * tf.math.log(prob), axis=-1)
  return tf.reduce_mean(neg_p_log_p_vals)


class Alpha(tf.Module):
  """Container for alpha, the architecture variable for weighting ops."""

  def __init__(self, arch_vars: List[tf.Variable],
               softmax_temperature: tf.Variable):
    """Initializer for Alpha.

    Args:
      arch_vars: List of architecture variables, with increasing sizes. One
        architecture variable is created for every intermediate node in the
        cell.
      softmax_temperature: Scalar tf.Variable for softmax temperature scaling.
        tf.Variable allows annealing if needed.
    """
    super().__init__()
    self.arch_vars = arch_vars
    self.softmax_temperature = softmax_temperature

  def total_mean_entropy(self) -> FloatLike:
    """Averages entropy across all MixedOps."""
    entropy_list = [
        arch_var_entropy(arch_var, self.softmax_temperature)
        for arch_var in self.arch_vars
    ]
    return tf.reduce_mean(tf.stack(entropy_list))

  def arch_probs(self) -> List[FloatLike]:
    return [
        tf.nn.softmax(self.softmax_temperature * arch_var, axis=-1)
        for arch_var in self.arch_vars
    ]


@pg.members([
    ('model_type',
     pg.typing.Enum(default=ModelType.CNN,
                    values=ModelType.list()), 'CNN or RNN search.'),
    ('output_mode',
     pg.typing.Enum(
         default=CellOutputMode.CONCAT_CONV,
         values=CellOutputMode.list()), 'Output Mode for the cell.'),
    ('use_batch_norm', pg.typing.Bool(default=True),
     'Whether to use batch normalization in the cell.'),
    ('reduction_stride', pg.typing.Int(default=1, min_value=1),
     'Image reduction size on the first op. Value of 1 means no reduction.'),
    ('num_inputs', pg.typing.Int(default=1), 'How many starting inputs.')
])
class CellConfig(pg.Object):
  """Base class for all cell configurations."""

  @property
  @abc.abstractmethod
  def num_nodes(self) -> int:
    pass

  @abc.abstractmethod
  def visualize_graph_url(self) -> str:
    """Returns a graphviz url for the cell."""
    pass


@pg.members([('cell_configs', pg.typing.Dict(), 'Dictionary of configs')])
class CellConfigDict(pg.Object):
  """Represents a dictionary of cell configs."""


CellConfigType = TypeVar('CellConfigType', bound=CellConfig)


class Cell(snt.Module):
  """Base class for cells. Contains common attributes and functions."""

  def __init__(self, output_channels: int, cell_config: CellConfigType,
               name: str):
    super().__init__(name)
    self._output_channels = output_channels
    self._cell_config = cell_config

    if self._cell_config.model_type == ModelType.CNN:
      self._preprocessing_layers = [
          snt.Conv2D(output_channels=output_channels, kernel_shape=1)
          for _ in range(self._cell_config.num_inputs)
      ]
    elif self._cell_config.model_type == ModelType.RNN:
      self._preprocessing_layers = [
          snt.Linear(output_size=output_channels)
          for _ in range(self._cell_config.num_inputs)
      ]

    if self._cell_config.output_mode == CellOutputMode.CONCAT_CONV:
      self._conv_reduction = snt.Conv2D(self._output_channels, kernel_shape=1)

    # BatchNorm is sometimes used for stability in DARTS.
    # Here we make sure no training variables are created.
    if self._cell_config.use_batch_norm:
      # Requires eps=1e-3, decay_rate=0.99 to avoid vanishing gradients.
      self._bn_normalizer = tf.keras.layers.BatchNormalization(trainable=False)

  def _maybe_preprocess_inputs(self, inputs):
    # Make sure inputs have the same channels as desired output channels.
    # This guarantees outputs from channel preserving ops (e.g. nonlinearity,
    # pooling) can be added with those from channel changing ops (e.g. conv).
    preprocessed_inputs = []
    for (x, layer) in zip(inputs, self._preprocessing_layers):
      if x.shape[-1] != self._output_channels:
        preprocessed_inputs.append(layer(x))
      else:
        preprocessed_inputs.append(x)

    return preprocessed_inputs

  def _output(self, internal_nodes, output_weights):
    if self._cell_config.output_mode == CellOutputMode.AVERAGE:
      return tf.add_n(internal_nodes) / float(len(internal_nodes))
    elif self._cell_config.output_mode == CellOutputMode.LAST_NODE:
      return internal_nodes[-1]
    elif self._cell_config.output_mode == CellOutputMode.SUM:
      return tf.add_n(internal_nodes)
    elif self._cell_config.output_mode == CellOutputMode.WEIGHTED_SUM:
      return tf.reduce_sum(
          tf.stack(internal_nodes, axis=-1) * output_weights, axis=-1)
    elif self._cell_config.output_mode == CellOutputMode.CONCAT_CONV:
      return self._conv_reduction(tf.concat(internal_nodes, axis=-1))
    else:
      raise ValueError(f'Unknown cell output mode: {self._cell_output_mode}')


# A list of input node id's and corresponding operator names.
FixedNodeConfig = pg.typing.List(
    pg.typing.Tuple([pg.typing.Int(), pg.typing.Str()]))


@pg.members([('fixed_node_configs', pg.typing.List(FixedNodeConfig), ''),
             ('output_weights',
              pg.typing.List(pg.typing.Float(min_value=0., max_value=1.)), '')])
class FixedCellConfig(CellConfig):
  """Cell configuration for discrete cells."""

  @property
  def num_nodes(self):
    return len(self.fixed_node_configs)

  def visualize_graph_url(self, aspect_ratio=None) -> str:
    graph = pydot.Dot(graph_type='digraph', rankdir='LR', ratio=aspect_ratio)
    # Style from https://github.com/quark0/darts/blob/master/cnn/visualize.py
    graph.set_node_defaults(
        style='filled',
        shape='rect',
        align='center',
        height='0.5',
        width='0.5',
        penwidth='2',
        fontname='times')
    graph.set_edge_defaults(fontname='times', fillcolor='gray')

    node0 = pydot.Node(0, label='input', fillcolor='darkseagreen2')
    # Input + internal nodes + output node.
    nodes = [node0] + [
        pydot.Node(i, label=f'{i}', fillcolor='lightblue')
        for i in range(1, self.num_nodes + 1)
    ] + [
        pydot.Node(
            self.num_nodes + 1, label='output', fillcolor='palegoldenrod')
    ]
    for n in nodes:
      graph.add_node(n)

    # Internal nodes.
    for i, node_config in enumerate(self.fixed_node_configs):
      for (ind, op) in node_config:
        graph.add_edge(pydot.Edge(nodes[ind], nodes[i + 1], label=op))

    cell_output_mode = CellOutputMode(self.output_mode)
    # Connect internal nodes to the output node.
    if cell_output_mode == CellOutputMode.LAST_NODE:
      graph.add_edge(pydot.Edge(nodes[-2], nodes[-1]))
    elif cell_output_mode in [CellOutputMode.SUM, CellOutputMode.CONCAT_CONV]:
      for j in range(self.num_nodes):
        graph.add_edge(pydot.Edge(nodes[j + 1], nodes[-1]))
    elif cell_output_mode == CellOutputMode.AVERAGE:
      w = 1.0 / float(self.num_nodes)
      for j in range(self.num_nodes):
        graph.add_edge(pydot.Edge(nodes[j + 1], nodes[-1], label=f'{w:.3f}'))
    elif cell_output_mode == CellOutputMode.WEIGHTED_SUM:
      for j in range(self.num_nodes):
        w = self.output_weights[j]
        graph.add_edge(pydot.Edge(nodes[j + 1], nodes[-1], label=f'{w:.3f}'))
    else:
      raise ValueError(f'Unknown cell output mode: {cell_output_mode}')

    escaped_dot_string = urllib.parse.quote_plus(graph.to_string())
    return f'{_GRAPHVIZ_URL}&dot={escaped_dot_string}'


@pg.members([('search_space', pg.typing.Object(darts_ops.SearchSpace),
              'Search Space for the cell.')])
class DartsCellConfig(CellConfig):
  """A DARTS/Dense cell config."""

  def __init__(self, *args, **kwargs):
    """Barebones init, since `pg.load` does not allow __init__ args."""
    super().__init__(*args, **kwargs)
    self._alpha = None  # This needs to be set externally.

  @property
  def alpha(self) -> Alpha:
    return self._alpha

  @alpha.setter
  def alpha(self, a: Alpha):
    self._alpha = a

  @classmethod
  def from_create_alpha(
      cls,
      op_names: List[str],
      num_nodes: int,
      output_mode: CellOutputMode,
      trainable: bool,
      softmax_temperature: float,
      trainable_temperature: bool = False,
      alpha_initializer=tf.zeros_initializer,
      arch_vars_name: Optional[str] = 'arch_vars',
      use_batch_norm: bool = True,
      reduction_stride: int = 1,
      num_inputs: int = 1,
      model_type: ModelType = ModelType.CNN) -> 'DartsCellConfig':
    """Generates Alpha and creates DartsCell class. Actual 'init' function."""
    arch_vars = []
    for i in range(num_nodes):
      arch_vars.append(
          tf.Variable(
              alpha_initializer()(shape=[num_inputs +
                                         i, len(op_names)]),
              trainable=trainable,
              name=f'{arch_vars_name}_{i}'))
    if output_mode == CellOutputMode.WEIGHTED_SUM:
      arch_vars.append(
          tf.Variable(
              alpha_initializer()(shape=[1, num_nodes]),
              trainable=trainable,
              name=f'{arch_vars_name}_output'))

    search_space = darts_ops.SearchSpace(op_names=op_names)
    alpha = Alpha(
        arch_vars=arch_vars,
        softmax_temperature=tf.Variable(
            softmax_temperature,
            trainable=trainable_temperature,
            name=f'{arch_vars_name}_softmax_temp'))
    darts_cell_config = cls(
        search_space=search_space,
        output_mode=output_mode,
        use_batch_norm=use_batch_norm,
        reduction_stride=reduction_stride,
        num_inputs=num_inputs,
        model_type=model_type)
    darts_cell_config.alpha = alpha
    return darts_cell_config

  def save(self, file_dir: str) -> None:
    tf.saved_model.save(self.alpha, file_dir)
    json_filepath = os.path.join(file_dir, 'darts_cell_config.json')
    pg.save(self, json_filepath)

  @classmethod
  def load(cls, file_dir: str) -> 'DartsCellConfig':
    loaded_alpha = tf.saved_model.load(file_dir)
    alpha = Alpha(
        arch_vars=loaded_alpha.arch_vars,
        softmax_temperature=loaded_alpha.softmax_temperature)
    json_filepath = os.path.join(file_dir, 'darts_cell_config.json')
    darts_cell_config_class = pg.load(json_filepath)
    darts_cell_config_class.alpha = alpha
    return darts_cell_config_class

  @property
  def num_ops(self) -> int:
    return len(self.search_space.op_names)

  @property
  def num_nodes(self) -> int:
    # Num of intermediate nodes
    if self.output_mode == CellOutputMode.WEIGHTED_SUM:
      return len(self.alpha.arch_vars) - 1
    else:
      return len(self.alpha.arch_vars)

  def to_fixed_cell_config(self, num_pred: int = 2) -> FixedCellConfig:
    """Converts a dense DARTS cell into a fixed one.

    Args:
      num_pred: Number of predecssors for each intermediate node.

    Returns:
      A `FixedCellConfig` object.
    """
    if self.output_mode == CellOutputMode.WEIGHTED_SUM:
      output_weights = tf.nn.softmax(
          self.alpha.softmax_temperature * self.alpha.arch_vars[-1],
          axis=-1).numpy()[0].tolist()
    else:
      output_weights = []

    arch_probs_tf = self.alpha.arch_probs()[:self.num_nodes]
    arch_probs_np = [arch_prob.numpy() for arch_prob in arch_probs_tf]

    # Drop zero operations by setting its probability to 0.
    if 'Zero' in self.search_space.op_names:
      ind = self.search_space.op_names.index('Zero')
      for prob in arch_probs_np:
        prob[:, ind] = 0

    # For each node, reduce each incoming MixedOp to a single best_op via
    # argmax selection, then keep `num_pred` strongest predecessors.
    best_ops = [a.argmax(-1) for a in arch_probs_np]
    predecessors = [
        np.argsort(-a.max(axis=-1))[:num_pred] for a in arch_probs_np
    ]

    node_configs = []
    for i in range(self.num_nodes):
      node_inputs = predecessors[i]
      # Record the op that connects input `j` to node `i`.
      node_config = [(int(j), self.search_space.op_names[best_ops[i][j]])
                     for j in node_inputs]
      node_configs.append(node_config)
    return FixedCellConfig(
        fixed_node_configs=node_configs,
        output_weights=output_weights,
        output_mode=self.output_mode,
        use_batch_norm=self.use_batch_norm,
        reduction_stride=self.reduction_stride,
        num_inputs=self.num_inputs,
        model_type=self.model_type)

  def visualize_graph_url(self) -> str:
    graph = pydot.Dot(graph_type='digraph', rankdir='LR')
    node0 = pydot.Node(0, label='input')
    # Input + internal nodes + output node.
    nodes = [node0] + [
        pydot.Node(i, label=f'node{i}') for i in range(1, self.num_nodes + 1)
    ] + [pydot.Node(self.num_nodes + 1, label='output')]
    for n in nodes:
      graph.add_node(n)

    # Make DARTS connections for internal nodes.
    for i, arch_var in enumerate(self.alpha.arch_vars[0:self.num_nodes]):
      # Make connections for nodes[i+1], which is the i-th internal node.
      arch_prob = tf.nn.softmax(
          self.alpha.softmax_temperature * arch_var, axis=-1)
      for j in range(len(arch_prob)):
        # Connect j-th node (including input) to i-th internal node.
        weights = arch_prob[j]
        assert len(weights) == len(
            self.search_space.op_names), ('weights size should match '
                                          'op_names.')
        for w, op in zip(weights, self.search_space.op_names):
          graph.add_edge(
              pydot.Edge(nodes[j], nodes[i + 1], label=f'{op}:{w:.3f}'))

    # Connect internal nodes to the output node.
    if self.output_mode == CellOutputMode.LAST_NODE:
      graph.add_edge(pydot.Edge(nodes[-2], nodes[-1], label='identity'))
    elif self.output_mode == CellOutputMode.SUM:
      for j in range(self.num_nodes):
        graph.add_edge(pydot.Edge(nodes[j + 1], nodes[-1], label='1.0'))
    elif self.output_mode == CellOutputMode.AVERAGE:
      w = 1.0 / float(self.num_nodes)
      for j in range(self.num_nodes):
        graph.add_edge(pydot.Edge(nodes[j + 1], nodes[-1], label=f'{w:.3f}'))
    elif self.output_mode == CellOutputMode.WEIGHTED_SUM:
      assert self.alpha.arch_vars[-1].shape[-1] == self.num_nodes
      arch_prob = tf.nn.softmax(
          self.alpha.softmax_temperature * self.alpha.arch_vars[-1], axis=-1)
      for j in range(self.num_nodes):
        w = arch_prob[0, j]
        graph.add_edge(pydot.Edge(nodes[j + 1], nodes[-1], label=f'{w:.3f}'))
    elif self.output_mode == CellOutputMode.CONCAT_CONV:
      for j in range(self.num_nodes):
        graph.add_edge(pydot.Edge(nodes[j + 1], nodes[-1], label='concat_conv'))
    else:
      raise ValueError(f'Unknown cell output mode: {self.output_mode}')

    escaped_dot_string = urllib.parse.quote_plus(graph.to_string())
    return f'{_GRAPHVIZ_URL}&dot={escaped_dot_string}'


class DartsCell(Cell):
  """A DARTS cell.

  A `DartsCell` takes `num_inputs` many inputs and produces an output.
  Internally, it has `num_nodes` intermediate nodes, and all of them are used to
  produce the cell output.

  To generate an intermediate node, we apply *individual* `MixedOp`s to all the
  inputs and all previous intermediate nodes, then sum the results. This is
  equation (1) in the DARTS paper.

  All these individual `MixedOp`s have similar forms (as specified by the
  `op_constructors`), but they use different architecture parameters to generate
  the mixture coefficients.

  NOTE: The architecture parameters are NOT owned by a `DartsCell`. They are
  owned by a higher level construct which may stack a few `DartsCell`s that
  share architecture parameters (but each cell learns its own weights for the
  Conv2D operator, etc.)
  """

  def __init__(self,
               output_channels: int,
               cell_config: DartsCellConfig,
               name: Optional[str] = 'DartsCell'):
    super().__init__(output_channels, cell_config, name=name)
    self._alpha = cell_config.alpha  # Needed for tracking.
    self._num_nodes = cell_config.num_nodes  # Needed for __call__.
    op_constructors = [
        darts_ops.OP_NAMES_TO_OP_CONSTRUCTORS[op_name]
        for op_name in self._cell_config.search_space.op_names
    ]

    # Use a list of lists to hold `MixedOp`s: each intermediate node requires
    # a list of `MixedOp`s.
    self._mixed_ops = [[] for _ in range(self._num_nodes)]
    for i in range(self._num_nodes):
      # Every intermediate node may connect to all the preceding nodes. Each
      # connection is represented by a MixedOp.
      for j in range(self._cell_config.num_inputs + i):
        # Only use custom stride if we are constructing a reduction cell and
        # this MixedOp connects to an input node.
        if j < self._cell_config.num_inputs:
          mixed_op_stride = self._cell_config.reduction_stride
        else:
          mixed_op_stride = 1
        self._mixed_ops[i].append(
            darts_ops.MixedOp(
                output_channels=output_channels,
                stride=mixed_op_stride,
                op_constructors=op_constructors))

  @property
  def alpha(self) -> Alpha:
    return self._alpha

  def _fwd_node(self, inputs, node_mixed_ops, node_arch_vars, is_training):
    """Forward computation of an intermediate node.

    Args:
      inputs: A list of inputs used to generate this internal node.
      node_mixed_ops: A list of `MixedOp`s.
      node_arch_vars: Node specific architecture parameters.
      is_training: Boolean flag for ops like BatchNorm and Dropout.

    Returns:
      The value of this intermediate node.
    """
    assert len(inputs) == len(node_mixed_ops)
    assert node_arch_vars.shape == [len(inputs), self._cell_config.num_ops]
    weights = tf.nn.softmax(
        self._alpha.softmax_temperature * node_arch_vars, axis=-1)
    results = []
    for i, (x, mixed_op) in enumerate(zip(inputs, node_mixed_ops)):
      results.append(mixed_op(x, weights[i], is_training=is_training))
    out = tf.add_n(results)
    if self._cell_config.use_batch_norm:
      # is_training is always True to apply BN without global moving averages.
      out = self._bn_normalizer(out, training=True)
    return out

  def __call__(self, inputs, is_training=True):
    """Computes cell output using cell specific architecture parameters."""
    assert len(inputs) == self._cell_config.num_inputs
    inputs = self._maybe_preprocess_inputs(inputs)
    # Check the input channels match output channels (of the conv op).
    # This is to make sure operations like x + conv(x) don't run into shape
    # issues. This assertion is guaranteed if the same conv op is used in a
    # preprocssing step (this is the case for IMPALA CNN).
    assert all([x.shape[-1] == self._output_channels for x in inputs])

    nodes = []
    for j in range(self._num_nodes):
      nodes.append(
          self._fwd_node(inputs + nodes, self._mixed_ops[j],
                         self._alpha.arch_vars[j], is_training))

    if self._cell_config.output_mode == CellOutputMode.WEIGHTED_SUM:
      output_weights = tf.nn.softmax(
          self._alpha.softmax_temperature *
          self._alpha.arch_vars[self._num_nodes],
          axis=-1)
    else:
      output_weights = None
    return self._output(nodes, output_weights)


class FixedCell(Cell):
  """Fixed cell (after discretization from a dense Darts cell)."""

  def __init__(self,
               output_channels: int,
               cell_config: FixedCellConfig,
               name: Optional[str] = 'FixedCell'):
    super().__init__(output_channels, cell_config, name)
    self._node_ops = [[] for _ in range(self._cell_config.num_nodes)]
    self._node_input_indexes = [None] * self._cell_config.num_nodes
    self._output_weights = tf.constant(self._cell_config.output_weights)

    # For each intermediate node, get its input indexes and the op for the
    # corresponding input.
    for i in range(self._cell_config.num_nodes):
      fixed_node_config = self._cell_config.fixed_node_configs[i]
      self._node_input_indexes[i] = [ind for (ind, _) in fixed_node_config]
      for (input_id, op_name) in fixed_node_config:
        if input_id < self._cell_config.num_inputs:
          op_stride = self._cell_config.reduction_stride
        else:
          op_stride = 1
        self._node_ops[i].append(darts_ops.OP_NAMES_TO_OP_CONSTRUCTORS[op_name](
            output_channels=output_channels, stride=op_stride))

  def _fwd_node(self, node_inputs, node_ops, is_training):
    assert len(node_inputs) == len(node_ops)
    results = [
        op(x, is_training=is_training)
        for (op, x) in zip(node_ops, node_inputs)
    ]
    out = tf.add_n(results)
    if self._cell_config.use_batch_norm:
      # is_training is always True to apply BN without global moving averages.
      out = self._bn_normalizer(out, training=True)
    return out

  def __call__(self, inputs, is_training=True):
    inputs = self._maybe_preprocess_inputs(inputs)
    nodes = []
    for i in range(self._cell_config.num_nodes):
      node_inputs = [(inputs + nodes)[j] for j in self._node_input_indexes[i]]
      nodes.append(self._fwd_node(node_inputs, self._node_ops[i], is_training))
    return self._output(nodes, self._output_weights)
