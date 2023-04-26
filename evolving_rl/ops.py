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

"""Node definitions for the computation graph."""
import abc
from typing import TypeVar

import pyglove as pg
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
from trfl import indexing_ops


def check_dtypes_same_type(input_dtypes):
  num_unique = len(set(input_dtypes))
  return num_unique == 1 or (num_unique == 2 and DTYPE.NONE in input_dtypes)


def check_dtypes_diff_type(input_dtypes):
  num_unique = len(set(input_dtypes))
  return num_unique == 2 or DTYPE.NONE in input_dtypes


# This size is used for initialized randomized inputs for doing duplicate
# program detection.
DEFAULT_SIZE = 4
# Range of float values used for randomized inputs for duplicate detection.
# Chosen as reasonable range of what to expect in environments.
RANDOM_FLOAT_RANGE = 100


def broadcast_to_larger(x, y):
  len_diff = abs(len(tf.shape(x)) - len(tf.shape(y)))
  if len_diff == 0:
    return x, y
  elif len(tf.shape(x)) > len(tf.shape(y)):
    return x, tf.reshape(y, y.shape + [1] * len_diff)
  else:
    return tf.reshape(x, x.shape + [1] * len_diff), y


class DTYPE(pg.Object):
  """Possible data types for computation graph.

  We assume that all vectors are the same dimension and that actions are
  discrete and so are indices.
  """
  FLOAT = 1
  VECTOR = 2  # These are fixed size
  STATE = 3
  ACTION = 4  # Actions are assumed to be discrete
  PARAM = 5  # Param that maps state to a list of real numbers
  NONE = 6
  LIST_ACTION_FLOAT = 7  # List of floats for all actions
  PROB = 8
  VAPARAM = 9  # Param that goes from vector to action
  FLOATCONSTANT = 10


@pg.members([
    ('input_idxs', pg.typing.List(pg.typing.Int(), default=[])),
    ('input_dtypes', pg.typing.List(pg.typing.Int(), default=[])),
    ('name', pg.typing.Str(default='default_name')),
])
class Node(pg.Object):
  """Node in computational graph.

  Will have a data type, value, and inputs.
  """

  def __str__(self):
    return self.name

  @abc.abstractmethod
  def set_output_dtype(self, input_dtypes):
    # Sets the output data type based on the input data type.
    pass


# Allows for typing of node class.
NodeConstructor = TypeVar('NodeConstructor', bound='Node')


class InputNode(Node):
  """Node for inputting existing values into graph."""

  def _on_bound(self):
    self.odtype = self.set_output_dtype(self.input_dtypes)

  def set_output_dtype(self, input_dtypes):
    return input_dtypes[0]

  def initialize_random_input(self, bs):
    dtype = self.odtype
    if dtype == DTYPE.FLOAT:
      return tf.random.uniform((bs, 1), -RANDOM_FLOAT_RANGE, RANDOM_FLOAT_RANGE)
    elif dtype == DTYPE.PARAM or dtype == DTYPE.VAPARAM:
      network = snt.Sequential([snt.nets.MLP([DEFAULT_SIZE, DEFAULT_SIZE])])
      return network
    elif dtype == DTYPE.ACTION:
      return tf.random.uniform((bs,),
                               0,
                               DEFAULT_SIZE - 1,
                               dtype=tf.dtypes.int32)
    elif dtype == DTYPE.STATE:
      return tf.random.uniform((bs, DEFAULT_SIZE), -100, 100)
    elif dtype == DTYPE.FLOATCONSTANT:
      return tf.random.uniform((bs, 1), -100, 100)
    else:
      raise NotImplementedError


class ParamNode(InputNode):
  """Parameter node contains neural network parameters."""

  def set_output_dtype(self, input_dtypes):
    return DTYPE.PARAM


class VAParamNode(InputNode):
  """Parameter node contains neural network parameters.

  These parameters will map vectors to actions.
  """

  def set_output_dtype(self, input_dtypes):
    return DTYPE.VAPARAM


class OpNode(Node):
  """Node for applying operations on previous nodes."""

  # Called after __init__
  def _on_bound(self):
    self.valid = self.check_valid(self.input_dtypes)
    # set internal dtype based on input dtypes
    self.odtype = self.set_output_dtype(self.input_dtypes)

  @property
  @abc.abstractmethod
  def num_inputs(self):
    pass

  @staticmethod
  @property
  def input_order_matters():
    return False

  @staticmethod
  @property
  def allowed_types():
    pass

  @abc.abstractmethod
  def execute(self, inputs):
    pass

  @abc.abstractmethod
  def check_valid(self, input_dtypes):
    # Checks if the operation is invalid after adding it to the graph.
    # This check is after the graph is defined and happens while the nodes are
    # being initialized (constructed) in order to prune out invalid ops based on
    # existing ops.
    pass

  @classmethod
  def precheck_valid_input(cls, nodes):
    # This check is while defining the graph but before the graph is actually
    # built in order to prune out invalid ops based on the input nodes.
    # The nodes will either be input nodes or dummy op nodes with none dtype.
    # A graph is defined with a ProgramSpec object and will be realized with
    # evolution. The graph definition through ProgramSpec will then be built
    # into a Program object which contains the instantiated nodes.
    return True


class DummyOpNode(OpNode):
  """Dummy operation for treating as intermediate nodes in the search space."""
  num_inputs = None

  def execute(self):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
    pass

  def check_valid(self, input_dtypes):
    return True

  def set_output_dtype(self, input_dtypes):
    return DTYPE.NONE

  def __str__(self):
    return 'DummyOp'


@pg.members([
    ('loss_weight', pg.typing.Float(1.)),
])
class LossOpNode(OpNode):
  """Apply loss to output."""
  num_inputs = 1
  input_order_matters = False
  allowed_types = set([DTYPE.FLOAT, DTYPE.NONE])
  loss_weights = [1.0, 0.5, 0.2, 0.1, 0.01, 0.001]

  def check_valid(self, input_dtypes):
    return input_dtypes[0] == DTYPE.FLOAT

  @classmethod
  def precheck_valid_input(cls, nodes):
    # This op will be added manually so always return False for this check.
    return False

  def set_output_dtype(self, input_dtypes):
    return input_dtypes[0]

  def execute(self, inputs):
    return self.loss_weight * inputs[0]

  def __str__(self):
    return 'Output * %.5f' % self.loss_weight


class SubtractOpNode(OpNode):
  """Subracts 2 nodes."""
  num_inputs = 2
  input_order_matters = True
  allowed_types = set(
      [DTYPE.FLOAT, DTYPE.VECTOR, DTYPE.LIST_ACTION_FLOAT, DTYPE.NONE])
  unallowed_type_pair = set([DTYPE.VECTOR, DTYPE.LIST_ACTION_FLOAT])

  def execute(self, inputs):
    x, y = broadcast_to_larger(inputs[0], inputs[1])
    return x - y

  def check_valid(self, input_dtypes):
    return set(input_dtypes).issubset(self.allowed_types)

  def set_output_dtype(self, input_dtypes):
    if DTYPE.LIST_ACTION_FLOAT in input_dtypes:
      return DTYPE.LIST_ACTION_FLOAT
    elif DTYPE.VECTOR in input_dtypes:
      return DTYPE.VECTOR
    else:
      return DTYPE.FLOAT

  @classmethod
  def precheck_valid_input(cls, nodes):
    dtypes = set([x.odtype for x in nodes])
    return dtypes.issubset(
        cls.allowed_types) and not dtypes.issubset(cls.unallowed_type_pair)

  def __str__(self):
    return '-'


class AddOpNode(OpNode):
  """General addition which allows vectors."""
  num_inputs = 2
  allowed_types = set(
      [DTYPE.FLOAT, DTYPE.VECTOR, DTYPE.LIST_ACTION_FLOAT, DTYPE.NONE])
  unallowed_type_pair = set([DTYPE.VECTOR, DTYPE.LIST_ACTION_FLOAT])

  def execute(self, inputs):
    x, y = broadcast_to_larger(inputs[0], inputs[1])
    return x + y

  def check_valid(self, input_dtypes):
    return set(input_dtypes).issubset(self.allowed_types)

  def set_output_dtype(self, input_dtypes):
    if DTYPE.LIST_ACTION_FLOAT in input_dtypes:
      return DTYPE.LIST_ACTION_FLOAT
    elif DTYPE.VECTOR in input_dtypes:
      return DTYPE.VECTOR
    else:
      return DTYPE.FLOAT

  @classmethod
  def precheck_valid_input(cls, nodes):
    dtypes = set([x.odtype for x in nodes])
    return dtypes.issubset(
        cls.allowed_types) and not dtypes.issubset(cls.unallowed_type_pair)

  def __str__(self):
    return '+'


class MultiplyFloatOpNode(OpNode):
  """Multiply a float with a different node type with broadcasting."""
  num_inputs = 2
  allowed_types = set([
      DTYPE.FLOAT, DTYPE.STATE, DTYPE.VECTOR, DTYPE.NONE,
      DTYPE.LIST_ACTION_FLOAT
  ])

  def execute(self, inputs):
    x, y = broadcast_to_larger(inputs[0], inputs[1])
    return x * y

  def check_valid(self, input_dtypes):
    return check_dtypes_diff_type(input_dtypes)

  def set_output_dtype(self, input_dtypes):
    if DTYPE.LIST_ACTION_FLOAT in input_dtypes:
      return DTYPE.LIST_ACTION_FLOAT
    elif DTYPE.VECTOR in input_dtypes:
      return DTYPE.VECTOR
    elif DTYPE.STATE in input_dtypes:
      return DTYPE.STATE
    else:
      return DTYPE.FLOAT

  @classmethod
  def precheck_valid_input(cls, nodes):
    dtypes = set([x.odtype for x in nodes])
    return dtypes.issubset(
        cls.allowed_types) and check_dtypes_diff_type(dtypes) and (
            DTYPE.FLOAT in dtypes or DTYPE.NONE in dtypes)

  def __str__(self):
    return '*'


class DotProductOpNode(OpNode):
  """Takes dot product."""
  num_inputs = 2
  allowed_types = set([
      DTYPE.FLOAT, DTYPE.STATE, DTYPE.VECTOR, DTYPE.NONE,
      DTYPE.LIST_ACTION_FLOAT
  ])

  def execute(self, inputs):
    x, y = broadcast_to_larger(inputs[0], inputs[1])
    tmp = x * y
    if self.input_dtypes[0] in [DTYPE.FLOAT, DTYPE.LIST_ACTION_FLOAT]:
      return tmp
    return tf.reduce_sum(tmp, axis=-1, keepdims=True)

  def check_valid(self, input_dtypes):
    return check_dtypes_same_type(input_dtypes)

  def set_output_dtype(self, input_dtypes):
    if DTYPE.LIST_ACTION_FLOAT in input_dtypes:
      return DTYPE.LIST_ACTION_FLOAT
    else:
      return DTYPE.FLOAT

  @classmethod
  def precheck_valid_input(cls, nodes):
    dtypes = set([x.odtype for x in nodes])
    return dtypes.issubset(cls.allowed_types) and check_dtypes_same_type(dtypes)

  def __str__(self):
    return '*'


class MaxOpNode(OpNode):
  """Takes max between nodes of the type."""
  num_inputs = 2
  allowed_types = set(
      [DTYPE.FLOAT, DTYPE.VECTOR, DTYPE.LIST_ACTION_FLOAT, DTYPE.NONE])

  def execute(self, inputs):
    x, y = broadcast_to_larger(inputs[0], inputs[1])
    return tf.math.maximum(x, y)

  def check_valid(self, input_dtypes):
    return check_dtypes_same_type(input_dtypes)

  def set_output_dtype(self, input_dtypes):
    return input_dtypes[0]

  @classmethod
  def precheck_valid_input(cls, nodes):
    dtypes = set([x.odtype for x in nodes])
    return dtypes.issubset(cls.allowed_types) and check_dtypes_same_type(dtypes)

  def __str__(self):
    return 'max'


class MaxFloatOpNode(OpNode):
  """Takes max of a float and different node type with broadcasting."""
  num_inputs = 2
  allowed_types = set(
      [DTYPE.FLOAT, DTYPE.VECTOR, DTYPE.LIST_ACTION_FLOAT, DTYPE.NONE])

  def execute(self, inputs):
    x, y = broadcast_to_larger(inputs[0], inputs[1])
    return tf.math.maximum(x, y)

  def check_valid(self, input_dtypes):
    return check_dtypes_diff_type(input_dtypes)

  def set_output_dtype(self, input_dtypes):
    if DTYPE.LIST_ACTION_FLOAT in input_dtypes:
      return DTYPE.LIST_ACTION_FLOAT
    elif DTYPE.VECTOR in input_dtypes:
      return DTYPE.VECTOR
    elif DTYPE.STATE in input_dtypes:
      return DTYPE.STATE
    else:
      return DTYPE.FLOAT

  @classmethod
  def precheck_valid_input(cls, nodes):
    dtypes = set([x.odtype for x in nodes])
    return dtypes.issubset(
        cls.allowed_types) and check_dtypes_diff_type(dtypes) and (
            DTYPE.FLOAT in dtypes or DTYPE.NONE in dtypes)

  def __str__(self):
    return 'maxf'


class MinOpNode(OpNode):
  """Takes min."""
  num_inputs = 2
  allowed_types = set(
      [DTYPE.FLOAT, DTYPE.VECTOR, DTYPE.LIST_ACTION_FLOAT, DTYPE.NONE])

  def execute(self, inputs):
    x, y = broadcast_to_larger(inputs[0], inputs[1])
    return tf.math.minimum(x, y)

  def check_valid(self, input_dtypes):
    return check_dtypes_same_type(input_dtypes)

  def set_output_dtype(self, input_dtypes):
    return input_dtypes[0]

  @classmethod
  def precheck_valid_input(cls, nodes):
    dtypes = set([x.odtype for x in nodes])
    return dtypes.issubset(cls.allowed_types)

  def __str__(self):
    return 'max'


class AddGaussianNoiseOpNode(OpNode):
  """Adds gaussian noise."""
  num_inputs = 1
  allowed_types = set([
      DTYPE.FLOAT, DTYPE.STATE, DTYPE.VECTOR, DTYPE.NONE,
      DTYPE.LIST_ACTION_FLOAT
  ])

  def execute(self, inputs):
    return tf.keras.layers.GaussianNoise(1)(inputs[0])

  def check_valid(self, input_dtypes):
    return input_dtypes[0] in self.allowed_types

  def set_output_dtype(self, input_dtypes):
    return input_dtypes[0]

  @classmethod
  def precheck_valid_input(cls, nodes):
    dtypes = set([x.odtype for x in nodes])
    return dtypes.issubset(cls.allowed_types)

  def __str__(self):
    return '+GaussianNoise'


class AbsOpNode(OpNode):
  """Take absolute value."""
  num_inputs = 1
  allowed_types = set([
      DTYPE.FLOAT, DTYPE.VECTOR, DTYPE.LIST_ACTION_FLOAT, DTYPE.STATE,
      DTYPE.NONE
  ])

  def execute(self, inputs):
    return tf.math.abs(inputs[0])

  def check_valid(self, input_dtypes):
    return input_dtypes[0] in self.allowed_types

  def set_output_dtype(self, input_dtypes):
    return input_dtypes[0]

  @classmethod
  def precheck_valid_input(cls, nodes):
    dtypes = set([x.odtype for x in nodes])
    return dtypes.issubset(cls.allowed_types)

  def __str__(self):
    return 'Abs'


class MultiplyTenthOpNode(OpNode):
  """Multiply value by 0.1."""
  num_inputs = 1
  allowed_types = set([DTYPE.FLOAT, DTYPE.NONE])

  def execute(self, inputs):
    return inputs[0] * 0.1

  def check_valid(self, input_dtypes):
    return input_dtypes[0] in self.allowed_types

  def set_output_dtype(self, input_dtypes):
    return input_dtypes[0]

  @classmethod
  def precheck_valid_input(cls, nodes):
    dtypes = set([x.odtype for x in nodes])
    return dtypes.issubset(cls.allowed_types)

  def __str__(self):
    return '* 0.1'


class DivOpNode(OpNode):
  """Divide 2 values."""
  num_inputs = 2
  input_order_matters = True
  allowed_types = set([
      DTYPE.FLOAT, DTYPE.VECTOR, DTYPE.LIST_ACTION_FLOAT, DTYPE.STATE,
      DTYPE.NONE
  ])

  def execute(self, inputs):
    return inputs[0] / (inputs[1] + 1e-10)

  def check_valid(self, input_dtypes):
    return check_dtypes_same_type(input_dtypes)

  def set_output_dtype(self, input_dtypes):
    return input_dtypes[0]

  @classmethod
  def precheck_valid_input(cls, nodes):
    dtypes = set([x.odtype for x in nodes])
    return dtypes.issubset(cls.allowed_types)

  def __str__(self):
    return '/'


class DivOpV2Node(OpNode):
  """Divide 2 values."""
  num_inputs = 2
  input_order_matters = True
  allowed_types = set([
      DTYPE.FLOAT, DTYPE.VECTOR, DTYPE.LIST_ACTION_FLOAT, DTYPE.STATE,
      DTYPE.NONE
  ])

  def execute(self, inputs):
    return inputs[0] / (inputs[1] + 1e-10)

  def check_valid(self, input_dtypes):
    return check_dtypes_same_type(input_dtypes)

  def set_output_dtype(self, input_dtypes):
    return input_dtypes[0]

  @classmethod
  def precheck_valid_input(cls, nodes):
    dtypes = set([x.odtype for x in nodes])
    return dtypes.issubset(cls.allowed_types) and check_dtypes_same_type(dtypes)

  def __str__(self):
    return '/'


class L2NormOpNode(OpNode):
  """Compute MSE. This node is mislabelled and should be renamed."""
  num_inputs = 1
  allowed_types = set([
      DTYPE.FLOAT, DTYPE.VECTOR, DTYPE.STATE, DTYPE.NONE,
      DTYPE.LIST_ACTION_FLOAT
  ])

  def check_valid(self, input_dtypes):
    return check_dtypes_same_type(input_dtypes)

  def set_output_dtype(self, input_dtypes):
    if DTYPE.LIST_ACTION_FLOAT in input_dtypes:
      return DTYPE.LIST_ACTION_FLOAT
    return DTYPE.FLOAT

  def execute(self, inputs):
    tmp = tf.math.square(inputs[0])
    if self.input_dtypes[0] in [DTYPE.FLOAT, DTYPE.LIST_ACTION_FLOAT]:
      return tmp
    return tf.reduce_sum(tmp, axis=-1, keepdims=True)

  @classmethod
  def precheck_valid_input(cls, nodes):
    dtypes = set([x.odtype for x in nodes])
    return dtypes.issubset(cls.allowed_types) and check_dtypes_same_type(dtypes)

  def __str__(self):
    return 'MSE'


class L2DistanceOpNode(OpNode):
  """Compute 0.5 * MSE. This node is mislabelled and should be renamed."""
  num_inputs = 2
  allowed_types = set([DTYPE.FLOAT, DTYPE.VECTOR, DTYPE.STATE, DTYPE.NONE])

  def check_valid(self, input_dtypes):
    return check_dtypes_same_type(input_dtypes)

  def set_output_dtype(self, input_dtypes):
    return DTYPE.FLOAT

  def execute(self, inputs):
    tmp = tf.math.square(inputs[0] - inputs[1])
    if self.input_dtypes[0] in [DTYPE.FLOAT]:
      return 0.5 * tmp
    return 0.5 * tf.reduce_sum(tmp, axis=-1, keepdims=True)

  @classmethod
  def precheck_valid_input(cls, nodes):
    dtypes = set([x.odtype for x in nodes])
    return dtypes.issubset(cls.allowed_types) and check_dtypes_same_type(dtypes)

  def __str__(self):
    return '1/2 MSE'


class SumListOpNode(OpNode):
  """Takes the sum of a list.

  Assumes input is (bs, dim1, ...) where dim1 is the axis to sum over.
  """
  num_inputs = 1
  allowed_types = set([DTYPE.LIST_ACTION_FLOAT, DTYPE.NONE])

  def check_valid(self, input_dtypes):
    return input_dtypes[0] in self.allowed_types

  @classmethod
  def precheck_valid_input(cls, nodes):
    dtypes = set([x.odtype for x in nodes])
    return dtypes.issubset(cls.allowed_types)

  def set_output_dtype(self, input_dtypes):
    return DTYPE.FLOAT

  def execute(self, inputs):
    return tf.reduce_sum(inputs[0], axis=1, keepdims=True)

  def __str__(self):
    return 'SumList'


class LogSumExpListOpNode(OpNode):
  """Takes the log sum exp of a list.

  Assumes input is (bs, dim1, ...) where dim1 is the axis to compute over.
  """
  num_inputs = 1
  allowed_types = set([DTYPE.LIST_ACTION_FLOAT, DTYPE.NONE])

  def check_valid(self, input_dtypes):
    return input_dtypes[0] in self.allowed_types

  @classmethod
  def precheck_valid_input(cls, nodes):
    dtypes = set([x.odtype for x in nodes])
    return dtypes.issubset(cls.allowed_types)

  def set_output_dtype(self, input_dtypes):
    return DTYPE.FLOAT

  def execute(self, inputs):
    return tf.math.reduce_logsumexp(inputs[0], axis=1, keepdims=True)

  def __str__(self):
    return 'LogSumExpList'


class MaxListOpNode(OpNode):
  """Takes the maximum value for a list.

  Assumes input is (bs, dim1, ...) where dim1 is the axis to maximize over.
  """
  num_inputs = 1
  allowed_types = set([DTYPE.LIST_ACTION_FLOAT, DTYPE.NONE])

  def check_valid(self, input_dtypes):
    return input_dtypes[0] in self.allowed_types

  @classmethod
  def precheck_valid_input(cls, nodes):
    dtypes = set([x.odtype for x in nodes])
    return dtypes.issubset(cls.allowed_types)

  def set_output_dtype(self, input_dtypes):
    return DTYPE.FLOAT

  def execute(self, inputs):
    return tf.reduce_max(inputs[0], axis=1, keepdims=True)

  def __str__(self):
    return 'MaxList'


class ArgMaxListOpNode(OpNode):
  """Returns the argument index which maximizes the value in a list.

  Assumes input is (bs, dim1, ...) where dim1 is the axis to maximize over.
  """
  num_inputs = 1
  allowed_types = set([DTYPE.LIST_ACTION_FLOAT, DTYPE.NONE])

  def check_valid(self, input_dtypes):
    return input_dtypes[0] in self.allowed_types

  @classmethod
  def precheck_valid_input(cls, nodes):
    dtypes = set([x.odtype for x in nodes])
    return dtypes.issubset(cls.allowed_types)

  def set_output_dtype(self, input_dtypes):
    return DTYPE.ACTION

  def execute(self, inputs):
    return tf.math.argmax(inputs[0], axis=1)

  def __str__(self):
    return 'ArgMaxList'


class SelectListOpNode(OpNode):
  """Selects value from list."""
  num_inputs = 2
  input_order_matters = True
  allowed_types = set([DTYPE.LIST_ACTION_FLOAT, DTYPE.ACTION, DTYPE.NONE])

  def check_valid(self, input_dtypes):
    return (input_dtypes[0] == DTYPE.LIST_ACTION_FLOAT and
            input_dtypes[1] == DTYPE.ACTION)

  @classmethod
  def precheck_valid_input(cls, nodes):
    return ((nodes[0].odtype == DTYPE.LIST_ACTION_FLOAT or
             nodes[0].odtype == DTYPE.NONE) and
            (nodes[1].odtype == DTYPE.ACTION or nodes[1].odtype == DTYPE.NONE))

  def set_output_dtype(self, input_dtypes):
    return DTYPE.FLOAT

  def execute(self, inputs):
    return indexing_ops.batched_index(
        tf.squeeze(inputs[0]), inputs[1], keepdims=True)

  def __str__(self):
    return 'SelectList'


class QValueListOpNode(OpNode):
  """Applies a q value network on a state to get q values for all actions.

  We assume that actions are discrete for now.
  """
  num_inputs = 2
  input_order_matters = True
  allowed_types = set([DTYPE.PARAM, DTYPE.STATE, DTYPE.NONE])

  def check_valid(self, input_dtypes):
    return input_dtypes[0] == DTYPE.PARAM and input_dtypes[1] == DTYPE.STATE

  @classmethod
  def precheck_valid_input(cls, nodes):
    return ((nodes[0].odtype == DTYPE.PARAM or
             nodes[0].odtype == DTYPE.NONE) and
            (nodes[1].odtype == DTYPE.STATE or nodes[1].odtype == DTYPE.NONE))

  def set_output_dtype(self, input_dtypes):
    return DTYPE.LIST_ACTION_FLOAT

  def execute(self, inputs):
    params, state = inputs
    q_vals = params(state)
    return q_vals

  def __str__(self):
    return 'QValueListOp'


class QValueListFromVecOpNode(OpNode):
  """Applies a q value network on a vector to get q values for all actions.

  We assume that actions are discrete for now.
  """
  num_inputs = 2
  input_order_matters = True
  allowed_types = set([DTYPE.VAPARAM, DTYPE.VECTOR, DTYPE.NONE])

  def check_valid(self, input_dtypes):
    return (input_dtypes[0] == DTYPE.VAPARAM and
            input_dtypes[1] == DTYPE.VECTOR)

  @classmethod
  def precheck_valid_input(cls, nodes):
    # Input nodes must be state, param, action.
    return ((nodes[0].odtype == DTYPE.VAPARAM or
             nodes[0].odtype == DTYPE.NONE) and
            (nodes[1].odtype == DTYPE.VECTOR or nodes[1].odtype == DTYPE.NONE))

  def set_output_dtype(self, input_dtypes):
    return DTYPE.LIST_ACTION_FLOAT

  def execute(self, inputs):
    params, state = inputs
    q_vals = params(state)
    return q_vals

  def __str__(self):
    return 'QValueListFromVecOp'


class StateEncoderOpNode(OpNode):
  """Encodes state to a vector."""
  num_inputs = 2
  input_order_matters = True
  allowed_types = set([DTYPE.PARAM, DTYPE.STATE, DTYPE.NONE])

  def check_valid(self, input_dtypes):
    return input_dtypes[0] == DTYPE.PARAM and input_dtypes[1] == DTYPE.STATE

  @classmethod
  def precheck_valid_input(cls, nodes):
    # Input nodes must be state, param, action.
    return ((nodes[0].odtype == DTYPE.PARAM or
             nodes[0].odtype == DTYPE.NONE) and
            (nodes[1].odtype == DTYPE.STATE or nodes[1].odtype == DTYPE.NONE))

  def set_output_dtype(self, input_dtypes):
    return DTYPE.VECTOR

  def execute(self, inputs):
    params, state = inputs
    encoding = params(state)
    return encoding

  def __str__(self):
    return 'StateEncoder'


class QValueOpNode(OpNode):
  """Applies a q value network on a state and action to get a q value.

  We assume that actions are discrete for now.
  """
  num_inputs = 3
  input_order_matters = True
  allowed_types = set([DTYPE.PARAM, DTYPE.STATE, DTYPE.ACTION, DTYPE.NONE])

  def check_valid(self, input_dtypes):
    return (input_dtypes[0] == DTYPE.PARAM and
            input_dtypes[1] == DTYPE.STATE and input_dtypes[2] == DTYPE.ACTION)

  @classmethod
  def precheck_valid_input(cls, nodes):
    # Input nodes must be state, param, action.
    return (nodes[0].odtype == DTYPE.PARAM and
            nodes[1].odtype == DTYPE.STATE and nodes[2].odtype == DTYPE.ACTION)

  def set_output_dtype(self, input_dtypes):
    return DTYPE.FLOAT

  def execute(self, inputs):
    params, state, action = inputs
    q_vals = params(state)
    q_val = indexing_ops.batched_index(
        tf.squeeze(q_vals), action, keepdims=True)
    return q_val

  def __str__(self):
    return 'QValueOp'


class SoftmaxOpNode(OpNode):
  """Apply softmax operator over function which outputs list."""
  num_inputs = 1
  input_order_matters = False
  allowed_types = set([DTYPE.LIST_ACTION_FLOAT, DTYPE.NONE])

  def check_valid(self, input_dtypes):
    return input_dtypes[0] == DTYPE.LIST_ACTION_FLOAT

  @classmethod
  def precheck_valid_input(cls, nodes):
    dtypes = set([x.odtype for x in nodes])
    return dtypes.issubset(cls.allowed_types)

  def set_output_dtype(self, input_dtypes):
    return DTYPE.PROB

  def execute(self, inputs):
    return tf.nn.softmax(inputs[0])

  def __str__(self):
    return 'Softmax'


class KLDivOpNode(OpNode):
  """Takes KL DIv."""
  num_inputs = 2
  input_order_matters = True
  allowed_types = set([DTYPE.PROB, DTYPE.NONE])

  def execute(self, inputs):
    kl_div = tf.keras.losses.KLDivergence(
        reduction=tf.keras.losses.Reduction.NONE)(inputs[0], inputs[1])
    return tf.reshape(kl_div, (-1, 1))

  def check_valid(self, input_dtypes):
    return check_dtypes_same_type(input_dtypes)

  def set_output_dtype(self, input_dtypes):
    return DTYPE.FLOAT

  @classmethod
  def precheck_valid_input(cls, nodes):
    dtypes = set([x.odtype for x in nodes])
    return dtypes.issubset(cls.allowed_types)

  def __str__(self):
    return 'KLDiv'


class EntropyOpNode(OpNode):
  """Compute entropy of probability distirubiton."""
  num_inputs = 1
  input_order_matters = False
  allowed_types = set([DTYPE.PROB, DTYPE.NONE])

  def check_valid(self, input_dtypes):
    return input_dtypes[0] == DTYPE.PROB

  @classmethod
  def precheck_valid_input(cls, nodes):
    dtypes = set([x.odtype for x in nodes])
    return dtypes.issubset(cls.allowed_types)

  def set_output_dtype(self, input_dtypes):
    return DTYPE.FLOAT

  def execute(self, inputs):
    return tf.math.reduce_sum(
        inputs[0] * tf.math.log(inputs[0] + 1e-8), axis=-1, keepdims=True)

  def __str__(self):
    return 'Entropy'


class LogOpNode(OpNode):
  """Take log of input."""
  num_inputs = 1
  input_order_matters = False
  allowed_types = set(
      [DTYPE.PROB, DTYPE.FLOAT, DTYPE.LIST_ACTION_FLOAT, DTYPE.NONE])

  def check_valid(self, input_dtypes):
    return input_dtypes[0] in self.allowed_types

  @classmethod
  def precheck_valid_input(cls, nodes):
    dtypes = set([x.odtype for x in nodes])
    return dtypes.issubset(cls.allowed_types)

  def set_output_dtype(self, input_dtypes):
    return input_dtypes[0]

  def execute(self, inputs):
    return tf.math.log(tf.math.abs(inputs[0]) + 1e-12)

  def __str__(self):
    return 'Log Abs'


class ExpOpNode(OpNode):
  """Compute exponential of input."""
  num_inputs = 1
  input_order_matters = False
  allowed_types = set(
      [DTYPE.PROB, DTYPE.FLOAT, DTYPE.LIST_ACTION_FLOAT, DTYPE.NONE])

  def check_valid(self, input_dtypes):
    return input_dtypes[0] in self.allowed_types

  @classmethod
  def precheck_valid_input(cls, nodes):
    dtypes = set([x.odtype for x in nodes])
    return dtypes.issubset(cls.allowed_types)

  def set_output_dtype(self, input_dtypes):
    return input_dtypes[0]

  def execute(self, inputs):
    return tf.clip_by_value(tf.math.exp(inputs[0]), -1e10, 1e10)

  def __str__(self):
    return 'Exp'


class ConstantGenNode(InputNode):
  """Node for generating constants."""

  def set_output_dtype(self, input_dtypes):
    return DTYPE.FLOATCONSTANT


class ConstantP5OpNode(OpNode):
  """Constant float value."""
  num_inputs = 1
  allowed_types = set([DTYPE.FLOATCONSTANT])

  def check_valid(self, input_dtypes):
    return input_dtypes[0] in self.allowed_types

  @classmethod
  def precheck_valid_input(cls, nodes):
    dtypes = set([x.odtype for x in nodes])
    return dtypes.issubset(cls.allowed_types)

  def set_output_dtype(self, input_dtypes):
    return DTYPE.FLOAT

  def execute(self, inputs):
    return tf.constant([[0.5]])

  def __str__(self):
    return '0.5'


class ConstantP2OpNode(OpNode):
  """Constant float value."""
  num_inputs = 1
  allowed_types = set([DTYPE.FLOATCONSTANT])

  def check_valid(self, input_dtypes):
    return input_dtypes[0] in self.allowed_types

  @classmethod
  def precheck_valid_input(cls, nodes):
    dtypes = set([x.odtype for x in nodes])
    return dtypes.issubset(cls.allowed_types)

  def set_output_dtype(self, input_dtypes):
    return DTYPE.FLOAT

  def execute(self, inputs):
    return tf.constant([[0.2]])

  def __str__(self):
    return '0.2'


class ConstantP1OpNode(OpNode):
  """Constant float value."""
  num_inputs = 1
  allowed_types = set([DTYPE.FLOATCONSTANT])

  def check_valid(self, input_dtypes):
    return input_dtypes[0] in self.allowed_types

  @classmethod
  def precheck_valid_input(cls, nodes):
    dtypes = set([x.odtype for x in nodes])
    return dtypes.issubset(cls.allowed_types)

  def set_output_dtype(self, input_dtypes):
    return DTYPE.FLOAT

  def execute(self, inputs):
    return tf.constant([[0.1]])

  def __str__(self):
    return '0.1'


class ConstantP01OpNode(OpNode):
  """Constant float value."""
  num_inputs = 1
  allowed_types = set([DTYPE.FLOATCONSTANT])

  def check_valid(self, input_dtypes):
    return input_dtypes[0] in self.allowed_types

  @classmethod
  def precheck_valid_input(cls, nodes):
    dtypes = set([x.odtype for x in nodes])
    return dtypes.issubset(cls.allowed_types)

  def set_output_dtype(self, input_dtypes):
    return DTYPE.FLOAT

  def execute(self, inputs):
    return tf.constant([[0.01]])

  def __str__(self):
    return '0.01'


class ConstantGaussianOpNode(OpNode):
  """Constant float value."""
  num_inputs = 1
  allowed_types = set([DTYPE.FLOATCONSTANT])

  def check_valid(self, input_dtypes):
    return input_dtypes[0] in self.allowed_types

  @classmethod
  def precheck_valid_input(cls, nodes):
    dtypes = set([x.odtype for x in nodes])
    return dtypes.issubset(cls.allowed_types)

  def set_output_dtype(self, input_dtypes):
    return DTYPE.FLOAT

  def execute(self, inputs):
    return tfp.distributions.Normal(loc=[[0]], scale=[1]).sample()

  def __str__(self):
    return 'Gaussian'


class ConstantUniformOpNode(OpNode):
  """Draws a value from the uniform distribution."""
  num_inputs = 1
  allowed_types = set([DTYPE.FLOATCONSTANT])

  def check_valid(self, input_dtypes):
    return input_dtypes[0] in self.allowed_types

  @classmethod
  def precheck_valid_input(cls, nodes):
    dtypes = set([x.odtype for x in nodes])
    return dtypes.issubset(cls.allowed_types)

  def set_output_dtype(self, input_dtypes):
    return DTYPE.FLOAT

  def execute(self, inputs):
    return tfp.distributions.Uniform(low=[[0]], high=[1]).sample()

  def __str__(self):
    return 'Uniform'


class MeanBatchOpNode(OpNode):
  """Computes the mean of an object on the batch or 1st dimension."""
  num_inputs = 1
  allowed_types = set([DTYPE.NONE, DTYPE.VECTOR, DTYPE.STATE, DTYPE.FLOAT])

  def check_valid(self, input_dtypes):
    return input_dtypes[0] in self.allowed_types

  @classmethod
  def precheck_valid_input(cls, nodes):
    dtypes = set([x.odtype for x in nodes])
    return dtypes.issubset(cls.allowed_types)

  def set_output_dtype(self, input_dtypes):
    return input_dtypes[0]

  def execute(self, inputs):
    return tf.math.reduce_mean(inputs[0], axis=0, keepdims=True)

  def __str__(self):
    return 'MeanBatch'


class MeanListOpNode(OpNode):
  """Compute the mean of a list."""
  num_inputs = 1
  allowed_types = set([DTYPE.NONE, DTYPE.LIST_ACTION_FLOAT])

  def check_valid(self, input_dtypes):
    return input_dtypes[0] in self.allowed_types

  @classmethod
  def precheck_valid_input(cls, nodes):
    dtypes = set([x.odtype for x in nodes])
    return dtypes.issubset(cls.allowed_types)

  def set_output_dtype(self, input_dtypes):
    return DTYPE.FLOAT

  def execute(self, inputs):
    return tf.math.reduce_mean(inputs[0], axis=1, keepdims=True)

  def __str__(self):
    return 'MeanList'


class StdBatchOpNode(OpNode):
  """Comptue the std dev across the batch size dimension."""
  num_inputs = 1
  allowed_types = set([DTYPE.NONE, DTYPE.VECTOR, DTYPE.STATE, DTYPE.FLOAT])

  def check_valid(self, input_dtypes):
    return input_dtypes[0] in self.allowed_types

  @classmethod
  def precheck_valid_input(cls, nodes):
    dtypes = set([x.odtype for x in nodes])
    return dtypes.issubset(cls.allowed_types)

  def set_output_dtype(self, input_dtypes):
    return input_dtypes[0]

  def execute(self, inputs):
    return tf.math.reduce_std(inputs[0], axis=0, keepdims=True)

  def __str__(self):
    return 'StdBatch'


class StdListOpNode(OpNode):
  """Compute the std dev of a list."""
  num_inputs = 1
  allowed_types = set([DTYPE.NONE, DTYPE.LIST_ACTION_FLOAT])

  def check_valid(self, input_dtypes):
    return input_dtypes[0] in self.allowed_types

  @classmethod
  def precheck_valid_input(cls, nodes):
    dtypes = set([x.odtype for x in nodes])
    return dtypes.issubset(cls.allowed_types)

  def set_output_dtype(self, input_dtypes):
    return DTYPE.FLOAT

  def execute(self, inputs):
    return tf.math.reduce_std(inputs[0], axis=1, keepdims=True)

  def __str__(self):
    return 'StdList'
