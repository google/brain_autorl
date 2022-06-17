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

"""Program definition for the computational graph."""
import itertools
import random
import traceback
from typing import List, Optional
import urllib

from absl import logging
from brain_autorl.evolving_rl.ops import ConstantGenNode
from brain_autorl.evolving_rl.ops import DTYPE
from brain_autorl.evolving_rl.ops import InputNode
from brain_autorl.evolving_rl.ops import LossOpNode
from brain_autorl.evolving_rl.ops import Node
from brain_autorl.evolving_rl.ops import NodeConstructor
from brain_autorl.evolving_rl.ops import OpNode
from brain_autorl.evolving_rl.program_search import ProgramSpec
import pydot
import tensorflow as tf


class Program():
  """Represents a program as a DAG."""

  def __init__(self, input_lst: List[Node], ops_lst: List[OpNode]):
    """Sets the program inputs and operationrs.

    Args:
      input_lst: list of nodes which specify inputs to the program
      ops_lst: list of nodes which specify the operations the program will
        execute in that order. Each node contains a list of idxs of its inputs.
        Will contain copy of input_lst as first segment.
    """
    self.input_lst = input_lst
    self.ops_lst = ops_lst

  @property
  def __name__(self):
    return 'program'

  def __call__(self, *args):
    # Executes the program.
    # Assumes args is in the same order as input_lst.

    # First just add input values to list so op nodes can index into them.
    try:
      values_lst = []
      for input_value in args:
        values_lst.append(input_value)

      # For generating constants. Assume last input is this node
      if len(values_lst) < len(self.input_lst) and isinstance(
          self.input_lst[-1], ConstantGenNode):
        values_lst.append(None)

      # Then compute all intermediate values of program which will operate over
      # nodes earlier in the list.
      for op in self.ops_lst:
        if not isinstance(op, InputNode):
          # print(op, op.input_idxs, len(values_lst))
          inputs = [values_lst[x] for x in op.input_idxs]
          values_lst.append(op.execute(inputs))
      return values_lst[-1]
    except Exception as e:
      logging.info('Exception: %s', traceback.format_exc())
      logging.info('Failed at op: %s', op)  # pytype: disable=name-error
      logging.info('Inputs: %s', inputs)  # pytype: disable=name-error
      logging.info('Input shapes: %s', [tf.shape(x) for x in inputs])  # pytype: disable=name-error
      logging.info('ops_lst: %s', self.ops_lst)
      logging.info('viz: %s', self.visualize())
      logging.info('Value list: %s', values_lst)
      raise e

  def check_path_exists(self, input_idx, output_idx):
    """Check path exists between input_idx and output_idx."""
    lst = []

    def check_node(node):
      if len(node.input_idxs) is None:
        return
      lst.extend(node.input_idxs)
      for x in node.input_idxs:
        check_node(self.ops_lst[x])

    check_node(self.ops_lst[output_idx])
    return input_idx in lst

  def find_leaf_nodes(self):
    """Find idxs of nodes that don't have any outputs."""
    input_idxs = []
    for op in self.ops_lst:
      input_idxs.extend(op.input_idxs)
    leaf_idxs = set(list(range(len(self.ops_lst)))) - set(input_idxs)
    # Remove input idxs.
    leaf_idxs = leaf_idxs - set(range(len(self.input_lst)))
    # print([self.ops_lst[x] for x in leaf_idxs])
    return leaf_idxs

  def visualize(self):
    """Visualize DAG with pydot."""
    # This only works for when running colab kernel locally. Borg does not work
    # with pydot create png.
    graph = pydot.Dot(graph_type='digraph')
    graph_op_dict = {}

    def get_fill_color(i, op):
      if i == len(self.ops_lst) - 1:
        return 'blue'
      elif op in self.input_lst:
        return 'green'
      else:
        return 'white'

    for i, op in enumerate(self.ops_lst):
      node = pydot.Node(
          i, label=str(op), style='filled', fillcolor=get_fill_color(i, op))
      graph.add_node(node)
      graph_op_dict[i] = node
      for j, input_idx in enumerate(op.input_idxs):
        if j > 0:
          color = 'red'  # Color 2nd edge red
        else:
          color = 'black'
        graph.add_edge(pydot.Edge(graph_op_dict[input_idx], node, color=color))

    graph_str = graph.to_string()
    return graph_str


def get_possible_input_idxs(op_cls, input_idxs):
  iter_func = itertools.combinations
  if op_cls.input_order_matters:
    iter_func = itertools.permutations
  return iter_func(input_idxs, op_cls.num_inputs)


def get_possible_ops_and_inputs(operators, all_inputs):
  """Compute all possible ops and inputs for that op.

  Args:
    operators: List of available operator types.
    all_inputs: All previous inputs and intermediate nodes in program which
      can be used an input for this op.

  Returns:
    A list of (Int, List[Int]) where first elem is index of operation
    and list if indexes of inputs to that operation.
  """

  possible_ops = list(set(operators) - set([LossOpNode]))
  valid_lst = []
  for op_cls in possible_ops:
    # Sample valid input idxs
    op_idx = operators.index(op_cls)
    for input_idxs in get_possible_input_idxs(op_cls, range(len(all_inputs))):
      input_ops = [all_inputs[idx] for idx in input_idxs]
      if op_cls.precheck_valid_input(input_ops):
        valid_lst.append((op_idx, list(input_idxs)))
  return valid_lst


def sample_valid_program_spec(input_nodes,
                              existing_ops,
                              program_length,
                              operators,
                              adjust_loss_weight=False):
  """Sample a valid program spec."""

  def sample_program_lst():
    all_inputs = input_nodes.copy()
    ops_cls_lst = []
    ops_cls_lst.extend(existing_ops)
    for op_idx, input_idxs in existing_ops:
      # import pdb; pdb.set_trace()
      # Find the data types for the op.
      input_dtypes = [all_inputs[idx].odtype for idx in input_idxs]
      # Construct the operator node with the specified input indices.
      op_cls = operators[op_idx]
      all_inputs.append(
          op_cls(input_idxs=input_idxs, input_dtypes=input_dtypes))

    for _ in range(program_length):
      possible_ops_and_inputs = get_possible_ops_and_inputs(
          operators, all_inputs)
      op_idx, input_idxs = random.choice(possible_ops_and_inputs)
      op_cls = operators[op_idx]

      # Add to lsts
      input_dtypes = [all_inputs[idx].odtype for idx in input_idxs]
      all_inputs.append(
          op_cls(input_idxs=input_idxs, input_dtypes=input_dtypes))
      ops_cls_lst.append((operators.index(op_cls), input_idxs))

    # Add output node which may multiply loss by constant
    ops_cls_lst.append((operators.index(LossOpNode), [len(all_inputs) - 1]))
    return ops_cls_lst

  while True:
    ops_cls_lst = sample_program_lst()
    loss_weight = random.choice(
        operators[-1].loss_weights) if adjust_loss_weight else 1.0
    program_spec = ProgramSpec(program_lst=ops_cls_lst, loss_weight=loss_weight)

    _, valid = build_program(input_nodes, program_spec, operators, 0)
    # print(program.visualize())

    if valid:
      return program_spec


def build_program(
    input_nodes: List[InputNode],
    program_spec: ProgramSpec,
    operators: List[NodeConstructor],
    check_path_diff: Optional[int] = None,
):
  """Builds a program from a list of input nodes and op class nodes.

  Args:
    input_nodes: input placeholders for program
    program_spec: specifies the operators and inputs to those operations in the
      program. The program_lst attribute of the program_spec will be a list of
      (int, List[int]) tuples where the first int is the idx of the operation
      in operators. The second list of integers is the indexes of the operation
      inputs. Indexing assumes that the input nodes are first, followed by the
      program_lst nodes so index of 0 corresponds to the first input node.
    operators: list of node class constructors specifying the available
      operations such as addition.
    check_path_diff: index of node to check that a path exists between the
      specified node and the output node.

  Returns:
   An instantiated Program.

  For example:
  input_nodes = [InputNode(name='Input A'), InputNode(name='Input B')]
  program_spec.program_lst = [(0, [0, 1], (1, [0, 2])]
  operators = [SubtractOpNode, DotProductOpNode]
  When the Program object is constructed, the ops_lst attribute will then be
    [(SubtractOpNode, (0, 1)), (DotProductOpNode, (0, 2))]
  The program would execute ops_lst in order.
  This means that the subtract op node will subtract Input A and Input B.
  The dot product op node will then multiply Input A with the result of the
  subtract op node.
  The output of the program is assumed to be the last operator node.
  """
  program_lst = program_spec.program_lst
  valid = True
  ops_lst = input_nodes.copy()
  input_idxs = None

  for op_idx, input_idxs in program_lst:
    # Find the data types for the op.
    input_dtypes = [ops_lst[idx].odtype for idx in input_idxs]
    # Construct the operator node with the specified input indices.
    op_cls = operators[op_idx]
    if op_cls == LossOpNode:
      op = op_cls(
          input_idxs=input_idxs,
          input_dtypes=input_dtypes,
          loss_weight=program_spec.loss_weight)
    else:
      op = op_cls(input_idxs=input_idxs, input_dtypes=input_dtypes)
    # Valid is checked and set after constructing the operation since we do not
    # know data types ahead of time so this is checked now.
    if not op.valid:
      valid = False
      logging.info('invalid OP')
      logging.info('%s, %s', op, [ops_lst[idx] for idx in input_idxs])
    ops_lst.append(op)
  # Assume last op is output node for now.
  # The last op data type must be a float to represent a loss function.
  if op.odtype != DTYPE.FLOAT:
    valid = False
    logging.info('Output OP is not FLOAT')
    logging.info('%s, %s', op, [ops_lst[idx] for idx in input_idxs])
  program = Program(input_nodes, ops_lst)

  # Check if path exists from output to this node.
  if check_path_diff is not None:
    if not program.check_path_exists(check_path_diff, len(ops_lst) - 1):
      valid = False
      logging.info('Program is not differentiable')
  return program, valid


class InvalidProgramError(Exception):
  pass
