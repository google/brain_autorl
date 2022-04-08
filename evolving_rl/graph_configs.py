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

"""Graph specifications for each experiment.

This file contains the graph specifications for each experiment. These
specifications include the available operators in the graph, number of nodes
in the graph, and what the existing operations in the graph are, and whether
to freeze the existing operations or not.
"""
import brain_autorl.evolving_rl.ops as opdefs
from brain_autorl.evolving_rl.ops import ConstantGenNode
from brain_autorl.evolving_rl.ops import DTYPE
from brain_autorl.evolving_rl.ops import InputNode
from brain_autorl.evolving_rl.ops import ParamNode
from brain_autorl.evolving_rl.program_search import create_search_space


def pre_graph_3(program_length=7):
  """Assume Q(s, a) is already added to graph."""

  ops = [
      opdefs.QValueListOpNode,
      opdefs.SubtractOpNode,
      opdefs.AddOpNode,
      opdefs.DotProductOpNode,
      opdefs.L2DistanceOpNode,
      opdefs.SelectListOpNode,
      opdefs.ArgMaxListOpNode,
      opdefs.MaxListOpNode,
      opdefs.SoftmaxOpNode,
      opdefs.EntropyOpNode,
      opdefs.LossOpNode,
  ]
  input_nodes = [
      ParamNode(name='q_param', input_dtypes=[DTYPE.PARAM]),
      InputNode(name='action', input_dtypes=[DTYPE.ACTION]),
      InputNode(name='s_tm1', input_dtypes=[DTYPE.STATE]),
      ParamNode(name='q_targ_param', input_dtypes=[DTYPE.PARAM]),
      InputNode(name='s_t', input_dtypes=[DTYPE.STATE]),
      InputNode(name='reward', input_dtypes=[DTYPE.FLOAT]),
      InputNode(name='discount', input_dtypes=[DTYPE.FLOAT])
  ]
  existing_ops = [
      (ops.index(opdefs.QValueListOpNode), [0, 2]),
      (ops.index(opdefs.SelectListOpNode), [len(input_nodes), 1]),
      (ops.index(opdefs.QValueListOpNode), [3, 4]),
      (ops.index(opdefs.MaxListOpNode), [len(input_nodes) + 2]),
  ]
  num_freeze_ops = len(existing_ops)
  search_space = create_search_space(input_nodes, existing_ops, program_length,
                                     ops)
  return ops, input_nodes, existing_ops, search_space, num_freeze_ops, program_length


def pre_graph_4(program_length=7):
  """Assume Q(s, a) is already added to graph and expand ops."""

  ops = [
      opdefs.QValueListOpNode,
      opdefs.SubtractOpNode,
      opdefs.AddOpNode,
      opdefs.DotProductOpNode,
      opdefs.L2DistanceOpNode,
      opdefs.SelectListOpNode,
      opdefs.ArgMaxListOpNode,
      opdefs.MaxListOpNode,
      opdefs.SoftmaxOpNode,
      opdefs.EntropyOpNode,
      opdefs.MinOpNode,
      opdefs.MaxOpNode,
      opdefs.AddGaussianNoiseOpNode,
      opdefs.KLDivOpNode,
      opdefs.LogOpNode,
      opdefs.ExpOpNode,
      opdefs.LossOpNode,
  ]

  input_nodes = [
      ParamNode(name='q_param', input_dtypes=[DTYPE.PARAM]),
      InputNode(name='action', input_dtypes=[DTYPE.ACTION]),
      InputNode(name='s_tm1', input_dtypes=[DTYPE.STATE]),
      ParamNode(name='q_targ_param', input_dtypes=[DTYPE.PARAM]),
      InputNode(name='s_t', input_dtypes=[DTYPE.STATE]),
      InputNode(name='reward', input_dtypes=[DTYPE.FLOAT]),
      InputNode(name='discount', input_dtypes=[DTYPE.FLOAT])
  ]
  existing_ops = [
      (ops.index(opdefs.QValueListOpNode), [0, 2]),
      (ops.index(opdefs.SelectListOpNode), [len(input_nodes), 1]),
      (ops.index(opdefs.QValueListOpNode), [3, 4]),
      (ops.index(opdefs.MaxListOpNode), [len(input_nodes) + 2]),
  ]
  num_freeze_ops = len(existing_ops)
  search_space = create_search_space(input_nodes, existing_ops, program_length,
                                     ops)
  return ops, input_nodes, existing_ops, search_space, num_freeze_ops, program_length


def pre_graph_5(program_length=8):
  """Assume Q(s, a) is already added to graph and expand ops."""

  ops = [
      opdefs.QValueListOpNode,
      opdefs.SubtractOpNode,
      opdefs.AddOpNode,
      opdefs.DotProductOpNode,
      opdefs.L2DistanceOpNode,
      opdefs.SelectListOpNode,
      opdefs.ArgMaxListOpNode,
      opdefs.MaxListOpNode,
      opdefs.SoftmaxOpNode,
      opdefs.EntropyOpNode,
      opdefs.MinOpNode,
      opdefs.MaxOpNode,
      opdefs.AddGaussianNoiseOpNode,
      opdefs.KLDivOpNode,
      opdefs.LogOpNode,
      opdefs.ExpOpNode,
      opdefs.AbsOpNode,
      opdefs.MultiplyTenthOpNode,
      opdefs.DivOpNode,
      opdefs.LossOpNode,
  ]
  input_nodes = [
      ParamNode(name='q_param', input_dtypes=[DTYPE.PARAM]),
      InputNode(name='action', input_dtypes=[DTYPE.ACTION]),
      InputNode(name='s_tm1', input_dtypes=[DTYPE.STATE]),
      ParamNode(name='q_targ_param', input_dtypes=[DTYPE.PARAM]),
      InputNode(name='s_t', input_dtypes=[DTYPE.STATE]),
      InputNode(name='reward', input_dtypes=[DTYPE.FLOAT]),
      InputNode(name='discount', input_dtypes=[DTYPE.FLOAT])
  ]
  existing_ops = [
      (ops.index(opdefs.QValueListOpNode), [0, 2]),
      (ops.index(opdefs.SelectListOpNode), [len(input_nodes), 1]),
      (ops.index(opdefs.QValueListOpNode), [3, 4]),
      (ops.index(opdefs.MaxListOpNode), [len(input_nodes) + 2]),
  ]
  num_freeze_ops = len(existing_ops)
  search_space = create_search_space(input_nodes, existing_ops, program_length,
                                     ops)
  return ops, input_nodes, existing_ops, search_space, num_freeze_ops, program_length


def pre_graph_5_existingdqn_nofreeze(program_length=14):
  """Assume Q(s, a) is already added to graph and do not freeze ops."""

  ops = [
      opdefs.QValueListOpNode,
      opdefs.SubtractOpNode,
      opdefs.AddOpNode,
      opdefs.DotProductOpNode,
      opdefs.L2DistanceOpNode,
      opdefs.SelectListOpNode,
      opdefs.ArgMaxListOpNode,
      opdefs.MaxListOpNode,
      opdefs.SoftmaxOpNode,
      opdefs.EntropyOpNode,
      opdefs.MinOpNode,
      opdefs.MaxOpNode,
      opdefs.AddGaussianNoiseOpNode,
      opdefs.KLDivOpNode,
      opdefs.LogOpNode,
      opdefs.ExpOpNode,
      opdefs.AbsOpNode,
      opdefs.MultiplyTenthOpNode,
      opdefs.DivOpNode,
      opdefs.LossOpNode,
  ]
  input_nodes = [
      ParamNode(name='q_param', input_dtypes=[DTYPE.PARAM]),
      InputNode(name='action', input_dtypes=[DTYPE.ACTION]),
      InputNode(name='s_tm1', input_dtypes=[DTYPE.STATE]),
      ParamNode(name='q_targ_param', input_dtypes=[DTYPE.PARAM]),
      InputNode(name='s_t', input_dtypes=[DTYPE.STATE]),
      InputNode(name='reward', input_dtypes=[DTYPE.FLOAT]),
      InputNode(name='discount', input_dtypes=[DTYPE.FLOAT])
  ]
  existing_ops = [(ops.index(opdefs.QValueListOpNode), [0, 2]),
                  (ops.index(opdefs.SelectListOpNode), [len(input_nodes), 1]),
                  (ops.index(opdefs.QValueListOpNode), [3, 4]),
                  (ops.index(opdefs.MaxListOpNode), [len(input_nodes) + 2])]
  existing_ops += [
      (ops.index(opdefs.DotProductOpNode), [len(input_nodes) - 1, 10]),
      (ops.index(opdefs.AddOpNode), [5, len(input_nodes) + len(existing_ops)]),
      (ops.index(opdefs.L2DistanceOpNode),
       [len(input_nodes) + 1,
        len(input_nodes) + len(existing_ops) + 1]),
  ]
  num_freeze_ops = 0
  search_space = create_search_space(
      input_nodes, existing_ops, program_length, ops, freeze_ops=False)
  return ops, input_nodes, existing_ops, search_space, num_freeze_ops, program_length


def pre_graph_6_existingdqn_nofreeze(program_length=14):
  """Assume Q(s, a) is already added to graph and do not freeze ops."""

  ops = [
      opdefs.QValueListOpNode,
      opdefs.SubtractOpNode,
      opdefs.AddOpNode,
      opdefs.DotProductOpNode,
      opdefs.L2DistanceOpNode,
      opdefs.SelectListOpNode,
      opdefs.ArgMaxListOpNode,
      opdefs.MaxListOpNode,
      opdefs.SoftmaxOpNode,
      opdefs.EntropyOpNode,
      opdefs.MinOpNode,
      opdefs.MaxOpNode,
      opdefs.AddGaussianNoiseOpNode,
      opdefs.KLDivOpNode,
      opdefs.LogOpNode,
      opdefs.ExpOpNode,
      opdefs.AbsOpNode,
      opdefs.MultiplyTenthOpNode,
      opdefs.DivOpNode,
      opdefs.LossOpNode,
  ]
  input_nodes = [
      ParamNode(name='q_param', input_dtypes=[DTYPE.PARAM]),
      InputNode(name='action', input_dtypes=[DTYPE.ACTION]),
      InputNode(name='s_tm1', input_dtypes=[DTYPE.STATE]),
      ParamNode(name='q_targ_param', input_dtypes=[DTYPE.PARAM]),
      InputNode(name='s_t', input_dtypes=[DTYPE.STATE]),
      InputNode(name='reward', input_dtypes=[DTYPE.FLOAT]),
      InputNode(name='discount', input_dtypes=[DTYPE.FLOAT])
  ]
  existing_ops = [(ops.index(opdefs.QValueListOpNode), [0, 2]),
                  (ops.index(opdefs.SelectListOpNode), [len(input_nodes), 1]),
                  (ops.index(opdefs.QValueListOpNode), [3, 4]),
                  (ops.index(opdefs.MaxListOpNode), [len(input_nodes) + 2])]
  existing_ops += [
      (ops.index(opdefs.DotProductOpNode), [len(input_nodes) - 1, 10]),
      (ops.index(opdefs.AddOpNode), [5, len(input_nodes) + len(existing_ops)]),
      (ops.index(opdefs.L2DistanceOpNode),
       [len(input_nodes) + 1,
        len(input_nodes) + len(existing_ops) + 1]),
  ]
  num_freeze_ops = 0
  search_space = create_search_space(
      input_nodes, existing_ops, program_length, ops, freeze_ops=False)
  return ops, input_nodes, existing_ops, search_space, num_freeze_ops, program_length


def pre_graph_7_existingdqn_nofreeze(program_length=14):
  """Assume Q(s, a) is added to graph and do not freeze ops. Add constants."""

  ops = [
      opdefs.QValueListOpNode,
      opdefs.SubtractOpNode,
      opdefs.AddOpNode,
      opdefs.DotProductOpNode,
      opdefs.L2DistanceOpNode,
      opdefs.SelectListOpNode,
      opdefs.MultiplyFloatOpNode,
      opdefs.ArgMaxListOpNode,
      opdefs.MaxListOpNode,
      opdefs.L2NormOpNode,
      opdefs.SoftmaxOpNode,
      opdefs.EntropyOpNode,
      opdefs.MinOpNode,
      opdefs.MaxOpNode,
      opdefs.MaxFloatOpNode,
      opdefs.AddGaussianNoiseOpNode,
      opdefs.KLDivOpNode,
      opdefs.SumListOpNode,
      opdefs.LogOpNode,
      opdefs.ExpOpNode,
      opdefs.AbsOpNode,
      opdefs.MultiplyTenthOpNode,
      opdefs.DivOpNode,
      opdefs.MeanBatchOpNode,
      opdefs.MeanListOpNode,
      opdefs.StdBatchOpNode,
      opdefs.StdListOpNode,
      opdefs.ConstantP5OpNode,
      opdefs.ConstantP2OpNode,
      opdefs.ConstantP1OpNode,
      opdefs.ConstantP01OpNode,
      opdefs.ConstantGaussianOpNode,
      opdefs.ConstantUniformOpNode,
      opdefs.LossOpNode,
  ]
  input_nodes = [
      ParamNode(name='q_param', input_dtypes=[DTYPE.PARAM]),
      InputNode(name='action', input_dtypes=[DTYPE.ACTION]),
      InputNode(name='s_tm1', input_dtypes=[DTYPE.STATE]),
      ParamNode(name='q_targ_param', input_dtypes=[DTYPE.PARAM]),
      InputNode(name='s_t', input_dtypes=[DTYPE.STATE]),
      InputNode(name='reward', input_dtypes=[DTYPE.FLOAT]),
      InputNode(name='discount', input_dtypes=[DTYPE.FLOAT]),
      ConstantGenNode(name='constant', input_dtypes=[DTYPE.FLOATCONSTANT])
  ]
  n_to_idx = {x.name: i for i, x in enumerate(input_nodes)}
  existing_ops = [
      (ops.index(opdefs.QValueListOpNode),
       [n_to_idx['q_param'], n_to_idx['s_tm1']]),
      (ops.index(opdefs.SelectListOpNode),
       [len(input_nodes), n_to_idx['action']]),
      (ops.index(opdefs.QValueListOpNode),
       [n_to_idx['q_targ_param'], n_to_idx['s_t']]),  # 10
      (ops.index(opdefs.MaxListOpNode), [len(input_nodes) + 2])
  ]
  existing_ops += [
      (ops.index(opdefs.DotProductOpNode),
       [n_to_idx['discount'],
        len(input_nodes) + len(existing_ops) - 1]),
      (ops.index(opdefs.AddOpNode), [5, len(input_nodes) + len(existing_ops)
                                    ]),  # 13
      (ops.index(opdefs.L2DistanceOpNode),
       [len(input_nodes) + 1,
        len(input_nodes) + len(existing_ops) + 1]),
  ]
  num_freeze_ops = 0
  search_space = create_search_space(
      input_nodes, existing_ops, program_length, ops, freeze_ops=False)
  return ops, input_nodes, existing_ops, search_space, num_freeze_ops, program_length


def pre_graph_9_existingdqn_nofreeze(program_length=13):
  """Assume Q(s, a) is already added to graph and do not freeze ops."""

  ops = [
      opdefs.QValueListOpNode,
      opdefs.SubtractOpNode,
      opdefs.AddOpNode,
      opdefs.DotProductOpNode,
      opdefs.L2DistanceOpNode,
      opdefs.SelectListOpNode,
      opdefs.MultiplyFloatOpNode,
      opdefs.ArgMaxListOpNode,
      opdefs.MaxListOpNode,
      opdefs.L2NormOpNode,
      opdefs.SoftmaxOpNode,
      opdefs.EntropyOpNode,
      opdefs.MinOpNode,
      opdefs.MaxOpNode,
      opdefs.MaxFloatOpNode,
      opdefs.AddGaussianNoiseOpNode,
      opdefs.KLDivOpNode,
      opdefs.SumListOpNode,
      opdefs.LogOpNode,
      opdefs.ExpOpNode,
      opdefs.AbsOpNode,
      opdefs.MultiplyTenthOpNode,
      opdefs.DivOpV2Node,
      opdefs.MeanBatchOpNode,
      opdefs.MeanListOpNode,
      opdefs.StdBatchOpNode,
      opdefs.StdListOpNode,
      opdefs.ConstantP5OpNode,
      opdefs.ConstantP2OpNode,
      opdefs.ConstantP1OpNode,
      opdefs.ConstantP01OpNode,
      opdefs.ConstantGaussianOpNode,
      opdefs.ConstantUniformOpNode,
      opdefs.LossOpNode,
  ]
  # self._network, a_tm1, o_tm1,
  # self._target_network, o_t, r_t, self._discount
  input_nodes = [
      ParamNode(name='q_param', input_dtypes=[DTYPE.PARAM]),
      InputNode(name='action', input_dtypes=[DTYPE.ACTION]),
      InputNode(name='s_tm1', input_dtypes=[DTYPE.STATE]),
      ParamNode(name='q_targ_param', input_dtypes=[DTYPE.PARAM]),
      InputNode(name='s_t', input_dtypes=[DTYPE.STATE]),
      InputNode(name='reward', input_dtypes=[DTYPE.FLOAT]),
      InputNode(name='discount', input_dtypes=[DTYPE.FLOAT]),
      ConstantGenNode(name='constant', input_dtypes=[DTYPE.FLOATCONSTANT])
  ]
  n_to_idx = {x.name: i for i, x in enumerate(input_nodes)}
  existing_ops = [
      (ops.index(opdefs.QValueListOpNode),
       [n_to_idx['q_param'], n_to_idx['s_tm1']]),
      (ops.index(opdefs.SelectListOpNode),
       [len(input_nodes), n_to_idx['action']]),
      (ops.index(opdefs.QValueListOpNode),
       [n_to_idx['q_targ_param'], n_to_idx['s_t']]),  # 10
      (ops.index(opdefs.MaxListOpNode), [len(input_nodes) + 2])
  ]
  existing_ops += [
      (ops.index(opdefs.ConstantP1OpNode), [n_to_idx['constant']]),
      (ops.index(opdefs.DotProductOpNode),
       [n_to_idx['discount'],
        len(input_nodes) + len(existing_ops) - 1]),
      (ops.index(opdefs.AddOpNode),
       [5, len(input_nodes) + len(existing_ops) + 1]),  # 13
      (ops.index(opdefs.L2DistanceOpNode),
       [len(input_nodes) + 1,
        len(input_nodes) + len(existing_ops) + 2]),
  ]
  num_freeze_ops = 0
  search_space = create_search_space(
      input_nodes, existing_ops, program_length, ops, freeze_ops=False)

  return ops, input_nodes, existing_ops, search_space, num_freeze_ops, program_length


def pre_graph_10_existingdqn_nofreeze(program_length=13):
  """Assume Q(s, a) is already added to graph and do not freeze ops."""

  ops = [
      opdefs.QValueListOpNode,
      opdefs.SubtractOpNode,
      opdefs.AddOpNode,
      opdefs.DotProductOpNode,
      opdefs.L2DistanceOpNode,
      opdefs.SelectListOpNode,
      opdefs.MultiplyFloatOpNode,
      opdefs.ArgMaxListOpNode,
      opdefs.MaxListOpNode,
      # opdefs.L2NormOpNode,
      opdefs.SoftmaxOpNode,
      opdefs.EntropyOpNode,
      opdefs.MinOpNode,
      opdefs.MaxOpNode,
      # opdefs.MaxFloatOpNode,
      opdefs.AddGaussianNoiseOpNode,
      opdefs.KLDivOpNode,
      opdefs.SumListOpNode,
      opdefs.LogOpNode,
      opdefs.ExpOpNode,
      opdefs.AbsOpNode,
      opdefs.MultiplyTenthOpNode,
      opdefs.DivOpV2Node,
      opdefs.MeanBatchOpNode,
      opdefs.MeanListOpNode,
      opdefs.StdBatchOpNode,
      opdefs.StdListOpNode,
      opdefs.ConstantP5OpNode,
      opdefs.ConstantP2OpNode,
      opdefs.ConstantP1OpNode,
      opdefs.ConstantP01OpNode,
      opdefs.ConstantGaussianOpNode,
      opdefs.ConstantUniformOpNode,
      opdefs.LossOpNode,
  ]
  input_nodes = [
      ParamNode(name='q_param', input_dtypes=[DTYPE.PARAM]),
      InputNode(name='action', input_dtypes=[DTYPE.ACTION]),
      InputNode(name='s_tm1', input_dtypes=[DTYPE.STATE]),
      ParamNode(name='q_targ_param', input_dtypes=[DTYPE.PARAM]),
      InputNode(name='s_t', input_dtypes=[DTYPE.STATE]),
      InputNode(name='reward', input_dtypes=[DTYPE.FLOAT]),
      InputNode(name='discount', input_dtypes=[DTYPE.FLOAT]),
      ConstantGenNode(name='constant', input_dtypes=[DTYPE.FLOATCONSTANT])
  ]
  n_to_idx = {x.name: i for i, x in enumerate(input_nodes)}
  existing_ops = [
      (ops.index(opdefs.QValueListOpNode),
       [n_to_idx['q_param'], n_to_idx['s_tm1']]),
      (ops.index(opdefs.SelectListOpNode),
       [len(input_nodes), n_to_idx['action']]),
      (ops.index(opdefs.QValueListOpNode),
       [n_to_idx['q_targ_param'], n_to_idx['s_t']]),  # 10
      (ops.index(opdefs.MaxListOpNode), [len(input_nodes) + 2])
  ]
  existing_ops += [
      (ops.index(opdefs.DotProductOpNode),
       [n_to_idx['discount'],
        len(input_nodes) + len(existing_ops) - 1]),
      (ops.index(opdefs.AddOpNode), [5, len(input_nodes) + len(existing_ops)
                                    ]),  # 13
      (ops.index(opdefs.L2DistanceOpNode),
       [len(input_nodes) + 1,
        len(input_nodes) + len(existing_ops) + +1]),
  ]
  num_freeze_ops = 0
  search_space = create_search_space(
      input_nodes, existing_ops, program_length, ops, freeze_ops=False)

  return ops, input_nodes, existing_ops, search_space, num_freeze_ops, program_length


def pre_graph_8_scratch(program_length=16):
  """Learn graph from scratch assuming few added ops."""

  ops = [
      opdefs.QValueListOpNode,
      opdefs.SubtractOpNode,
      opdefs.AddOpNode,
      opdefs.DotProductOpNode,
      opdefs.L2DistanceOpNode,
      opdefs.SelectListOpNode,
      opdefs.MultiplyFloatOpNode,
      opdefs.ArgMaxListOpNode,
      opdefs.MaxListOpNode,
      opdefs.L2NormOpNode,
      opdefs.SoftmaxOpNode,
      opdefs.EntropyOpNode,
      opdefs.MinOpNode,
      opdefs.MaxOpNode,
      opdefs.MaxFloatOpNode,
      opdefs.AddGaussianNoiseOpNode,
      opdefs.KLDivOpNode,
      opdefs.SumListOpNode,
      opdefs.LogOpNode,
      opdefs.ExpOpNode,
      opdefs.AbsOpNode,
      opdefs.MultiplyTenthOpNode,
      opdefs.DivOpV2Node,
      opdefs.MeanBatchOpNode,
      opdefs.MeanListOpNode,
      opdefs.StdBatchOpNode,
      opdefs.StdListOpNode,
      opdefs.ConstantP5OpNode,
      opdefs.ConstantP2OpNode,
      opdefs.ConstantP1OpNode,
      opdefs.ConstantP01OpNode,
      opdefs.ConstantGaussianOpNode,
      opdefs.ConstantUniformOpNode,
      opdefs.LossOpNode,
  ]
  input_nodes = [
      ParamNode(name='q_param', input_dtypes=[DTYPE.PARAM]),
      InputNode(name='action', input_dtypes=[DTYPE.ACTION]),
      InputNode(name='s_tm1', input_dtypes=[DTYPE.STATE]),
      ParamNode(name='q_targ_param', input_dtypes=[DTYPE.PARAM]),
      InputNode(name='s_t', input_dtypes=[DTYPE.STATE]),
      InputNode(name='reward', input_dtypes=[DTYPE.FLOAT]),
      InputNode(name='discount', input_dtypes=[DTYPE.FLOAT]),
      ConstantGenNode(name='constant', input_dtypes=[DTYPE.FLOATCONSTANT])
  ]
  n_to_idx = {x.name: i for i, x in enumerate(input_nodes)}
  existing_ops = [
      (ops.index(opdefs.QValueListOpNode),
       [n_to_idx['q_param'], n_to_idx['s_tm1']]),
      (ops.index(opdefs.SelectListOpNode),
       [len(input_nodes), n_to_idx['action']]),
  ]

  num_freeze_ops = 2
  search_space = create_search_space(
      input_nodes, existing_ops, program_length, ops, freeze_ops=True)

  return ops, input_nodes, existing_ops, search_space, num_freeze_ops, program_length


def pre_graph_11_scratch(program_length=16):
  """Learn graph from scratch assuming few added ops."""

  ops = [
      opdefs.QValueListOpNode,
      opdefs.SubtractOpNode,
      opdefs.AddOpNode,
      opdefs.DotProductOpNode,
      opdefs.L2DistanceOpNode,
      opdefs.SelectListOpNode,
      opdefs.MultiplyFloatOpNode,
      opdefs.ArgMaxListOpNode,
      opdefs.MaxListOpNode,
      # opdefs.L2NormOpNode,
      opdefs.SoftmaxOpNode,
      opdefs.EntropyOpNode,
      opdefs.MinOpNode,
      opdefs.MaxOpNode,
      # opdefs.MaxFloatOpNode,
      opdefs.AddGaussianNoiseOpNode,
      opdefs.KLDivOpNode,
      opdefs.SumListOpNode,
      opdefs.LogOpNode,
      opdefs.ExpOpNode,
      opdefs.AbsOpNode,
      opdefs.MultiplyTenthOpNode,
      opdefs.DivOpV2Node,
      opdefs.MeanBatchOpNode,
      opdefs.MeanListOpNode,
      opdefs.StdBatchOpNode,
      opdefs.StdListOpNode,
      opdefs.ConstantP5OpNode,
      opdefs.ConstantP2OpNode,
      opdefs.ConstantP1OpNode,
      opdefs.ConstantP01OpNode,
      opdefs.ConstantGaussianOpNode,
      opdefs.ConstantUniformOpNode,
      opdefs.LossOpNode,
  ]

  input_nodes = [
      ParamNode(name='q_param', input_dtypes=[DTYPE.PARAM]),
      InputNode(name='action', input_dtypes=[DTYPE.ACTION]),
      InputNode(name='s_tm1', input_dtypes=[DTYPE.STATE]),
      ParamNode(name='q_targ_param', input_dtypes=[DTYPE.PARAM]),
      InputNode(name='s_t', input_dtypes=[DTYPE.STATE]),
      InputNode(name='reward', input_dtypes=[DTYPE.FLOAT]),
      InputNode(name='discount', input_dtypes=[DTYPE.FLOAT]),
      ConstantGenNode(name='constant', input_dtypes=[DTYPE.FLOATCONSTANT])
  ]
  n_to_idx = {x.name: i for i, x in enumerate(input_nodes)}
  existing_ops = [
      (ops.index(opdefs.QValueListOpNode),
       [n_to_idx['q_param'], n_to_idx['s_tm1']]),
      (ops.index(opdefs.SelectListOpNode),
       [len(input_nodes), n_to_idx['action']]),
  ]
  num_freeze_ops = 2
  search_space = create_search_space(
      input_nodes, existing_ops, program_length, ops, freeze_ops=True)

  return ops, input_nodes, existing_ops, search_space, num_freeze_ops, program_length


def normal_dqn_config():
  """Graph for normal dqn."""

  ops = [
      opdefs.QValueListOpNode, opdefs.SubtractOpNode, opdefs.AddOpNode,
      opdefs.DotProductOpNode, opdefs.L2DistanceOpNode, opdefs.SelectListOpNode,
      opdefs.ArgMaxListOpNode, opdefs.MaxListOpNode, opdefs.LossOpNode
  ]
  input_nodes = [
      ParamNode(name='q_param', input_dtypes=[DTYPE.PARAM]),
      InputNode(name='action', input_dtypes=[DTYPE.ACTION]),
      InputNode(name='s_tm1', input_dtypes=[DTYPE.STATE]),
      ParamNode(name='q_targ_param', input_dtypes=[DTYPE.PARAM]),
      InputNode(name='s_t', input_dtypes=[DTYPE.STATE]),
      InputNode(name='reward', input_dtypes=[DTYPE.FLOAT]),
      InputNode(name='discount', input_dtypes=[DTYPE.FLOAT])
  ]
  existing_ops = [
      (ops.index(opdefs.QValueListOpNode), [0, 2]),
      (ops.index(opdefs.SelectListOpNode), [len(input_nodes), 1]),
      (ops.index(opdefs.QValueListOpNode), [3, 4]),
      (ops.index(opdefs.MaxListOpNode), [len(input_nodes) + 2]),
  ]
  num_freeze_ops = len(existing_ops)
  normal_dqn = existing_ops + [
      (ops.index(opdefs.DotProductOpNode), [len(input_nodes) - 1, 10]),
      (ops.index(opdefs.AddOpNode), [5, len(input_nodes) + len(existing_ops)]),
      (ops.index(opdefs.L2DistanceOpNode),
       [len(input_nodes) + 1,
        len(input_nodes) + len(existing_ops) + 1])
  ]
  program_length = 4
  search_space = create_search_space(input_nodes, existing_ops, program_length,
                                     ops)
  return ops, input_nodes, normal_dqn, search_space, num_freeze_ops, program_length


def dqn_reg_config():
  """Graph for DQN_Reg."""

  ops = [
      opdefs.QValueListOpNode, opdefs.SubtractOpNode, opdefs.AddOpNode,
      opdefs.DotProductOpNode, opdefs.L2DistanceOpNode, opdefs.SelectListOpNode,
      opdefs.MultiplyTenthOpNode, opdefs.ArgMaxListOpNode, opdefs.MaxListOpNode,
      opdefs.LossOpNode
  ]
  input_nodes = [
      ParamNode(name='q_param', input_dtypes=[DTYPE.PARAM]),
      InputNode(name='action', input_dtypes=[DTYPE.ACTION]),
      InputNode(name='s_tm1', input_dtypes=[DTYPE.STATE]),
      ParamNode(name='q_targ_param', input_dtypes=[DTYPE.PARAM]),
      InputNode(name='s_t', input_dtypes=[DTYPE.STATE]),
      InputNode(name='reward', input_dtypes=[DTYPE.FLOAT]),
      InputNode(name='discount', input_dtypes=[DTYPE.FLOAT])
  ]
  existing_ops = [
      (ops.index(opdefs.QValueListOpNode), [0, 2]),
      (ops.index(opdefs.SelectListOpNode), [len(input_nodes), 1]),
      (ops.index(opdefs.QValueListOpNode), [3, 4]),
      (ops.index(opdefs.MaxListOpNode), [len(input_nodes) + 2]),
  ]
  num_freeze_ops = len(existing_ops)
  normal_dqn = existing_ops + [
      (ops.index(opdefs.DotProductOpNode), [len(input_nodes) - 1, 10]),
      (ops.index(opdefs.AddOpNode), [5, len(input_nodes) + len(existing_ops)]),
      (ops.index(opdefs.L2DistanceOpNode),
       [len(input_nodes) + 1,
        len(input_nodes) + len(existing_ops) + 1]),
      (ops.index(opdefs.MultiplyTenthOpNode), [len(input_nodes) + 1]),
      (ops.index(opdefs.AddOpNode), [
          len(input_nodes) + len(existing_ops) + 2,
          len(input_nodes) + len(existing_ops) + 3
      ])
  ]
  program_length = 0
  search_space = create_search_space(input_nodes, existing_ops, program_length,
                                     ops)
  return ops, input_nodes, normal_dqn, search_space, num_freeze_ops, program_length


def dqn_reg_abs_config():
  """Graph for DQN_reg with absolute value on reg term."""

  ops = [
      opdefs.QValueListOpNode, opdefs.SubtractOpNode, opdefs.AddOpNode,
      opdefs.DotProductOpNode, opdefs.L2DistanceOpNode, opdefs.SelectListOpNode,
      opdefs.AbsOpNode, opdefs.MultiplyTenthOpNode, opdefs.ArgMaxListOpNode,
      opdefs.MaxListOpNode, opdefs.LossOpNode
  ]

  input_nodes = [
      ParamNode(name='q_param', input_dtypes=[DTYPE.PARAM]),
      InputNode(name='action', input_dtypes=[DTYPE.ACTION]),
      InputNode(name='s_tm1', input_dtypes=[DTYPE.STATE]),
      ParamNode(name='q_targ_param', input_dtypes=[DTYPE.PARAM]),
      InputNode(name='s_t', input_dtypes=[DTYPE.STATE]),
      InputNode(name='reward', input_dtypes=[DTYPE.FLOAT]),
      InputNode(name='discount', input_dtypes=[DTYPE.FLOAT])
  ]
  existing_ops = [
      (ops.index(opdefs.QValueListOpNode), [0, 2]),
      (ops.index(opdefs.SelectListOpNode), [len(input_nodes), 1]),
      (ops.index(opdefs.QValueListOpNode), [3, 4]),
      (ops.index(opdefs.MaxListOpNode), [len(input_nodes) + 2]),
  ]
  num_freeze_ops = len(existing_ops)
  normal_dqn = existing_ops + [
      (ops.index(opdefs.DotProductOpNode), [len(input_nodes) - 1, 10]),
      (ops.index(opdefs.AddOpNode), [5, len(input_nodes) + len(existing_ops)]),
      (ops.index(opdefs.L2DistanceOpNode),
       [len(input_nodes) + 1,
        len(input_nodes) + len(existing_ops) + 1]),
      (ops.index(opdefs.MultiplyTenthOpNode), [len(input_nodes) + 1]),
      (ops.index(opdefs.AbsOpNode), [len(input_nodes) + len(existing_ops) + 3]),
      (ops.index(opdefs.AddOpNode), [
          len(input_nodes) + len(existing_ops) + 2,
          len(input_nodes) + len(existing_ops) + 4
      ])
  ]
  program_length = 0
  search_space = create_search_space(input_nodes, existing_ops, program_length,
                                     ops)
  return ops, input_nodes, normal_dqn, search_space, num_freeze_ops, program_length


def cql_config():
  """Graph for conservative q learning (cql)."""

  ops = [
      opdefs.QValueListOpNode, opdefs.SubtractOpNode, opdefs.AddOpNode,
      opdefs.DotProductOpNode, opdefs.L2DistanceOpNode, opdefs.SelectListOpNode,
      opdefs.MultiplyTenthOpNode, opdefs.SoftmaxOpNode,
      opdefs.LogSumExpListOpNode, opdefs.SumListOpNode, opdefs.LogOpNode,
      opdefs.ExpOpNode, opdefs.ArgMaxListOpNode, opdefs.MaxListOpNode,
      opdefs.LossOpNode
  ]

  input_nodes = [
      ParamNode(name='q_param', input_dtypes=[DTYPE.PARAM]),
      InputNode(name='action', input_dtypes=[DTYPE.ACTION]),
      InputNode(name='s_tm1', input_dtypes=[DTYPE.STATE]),
      ParamNode(name='q_targ_param', input_dtypes=[DTYPE.PARAM]),
      InputNode(name='s_t', input_dtypes=[DTYPE.STATE]),
      InputNode(name='reward', input_dtypes=[DTYPE.FLOAT]),
      InputNode(name='discount', input_dtypes=[DTYPE.FLOAT])
  ]
  existing_ops = [
      (ops.index(opdefs.QValueListOpNode), [0, 2]),
      (ops.index(opdefs.SelectListOpNode), [len(input_nodes), 1]),
      (ops.index(opdefs.QValueListOpNode), [3, 4]),
      (ops.index(opdefs.MaxListOpNode), [len(input_nodes) + 2]),
  ]
  num_freeze_ops = len(existing_ops)
  si = len(input_nodes) + len(existing_ops)
  normal_dqn = existing_ops + [
      (ops.index(opdefs.DotProductOpNode), [len(input_nodes) - 1, 10]),
      (ops.index(opdefs.AddOpNode), [5, len(input_nodes) + len(existing_ops)]),
      (ops.index(opdefs.L2DistanceOpNode), [len(input_nodes) + 1, si + 1]),
      (ops.index(opdefs.LogSumExpListOpNode), [len(input_nodes)]),
      (ops.index(opdefs.SubtractOpNode), [si + 3, len(input_nodes) + 1]),
      (ops.index(opdefs.MultiplyTenthOpNode), [si + 4]),
      (ops.index(opdefs.AddOpNode), [si + 2, si + 5])
      # (ops.index(opdefs.AddOpNode), [si + 2, si+4])
  ]
  program_length = 0
  search_space = create_search_space(input_nodes, existing_ops, program_length,
                                     ops)
  return ops, input_nodes, normal_dqn, search_space, num_freeze_ops, program_length


def ddqn_config():
  """Graph for double dqn."""

  ops = [
      opdefs.QValueListOpNode,
      opdefs.SubtractOpNode,
      opdefs.AddOpNode,
      opdefs.DotProductOpNode,
      opdefs.L2DistanceOpNode,
      opdefs.SelectListOpNode,
      opdefs.ArgMaxListOpNode,
      opdefs.MaxListOpNode,
      opdefs.LossOpNode,
  ]
  # self._network, a_tm1, o_tm1,
  # self._target_network, o_t, r_t, self._discount
  input_nodes = [
      ParamNode(name='q_param', input_dtypes=[DTYPE.PARAM]),
      InputNode(name='action', input_dtypes=[DTYPE.ACTION]),
      InputNode(name='s_tm1', input_dtypes=[DTYPE.STATE]),
      ParamNode(name='q_targ_param', input_dtypes=[DTYPE.PARAM]),
      InputNode(name='s_t', input_dtypes=[DTYPE.STATE]),
      InputNode(name='reward', input_dtypes=[DTYPE.FLOAT]),
      InputNode(name='discount', input_dtypes=[DTYPE.FLOAT])
  ]
  existing_ops = [
      (ops.index(opdefs.QValueListOpNode), [0, 2]),
      (ops.index(opdefs.SelectListOpNode), [len(input_nodes), 1]),
      (ops.index(opdefs.QValueListOpNode), [0, 4]),
      (ops.index(opdefs.ArgMaxListOpNode), [len(input_nodes) + 2]),
      (ops.index(opdefs.QValueListOpNode), [3, 4]),
      (ops.index(opdefs.SelectListOpNode),
       [len(input_nodes) + 4, len(input_nodes) + 3]),
  ]
  num_freeze_ops = len(existing_ops)
  ddqn = existing_ops + [
      (ops.index(opdefs.DotProductOpNode),
       [len(input_nodes) - 1,
        len(input_nodes) + len(existing_ops) - 1]),
      (ops.index(opdefs.AddOpNode), [5, len(input_nodes) + len(existing_ops)]),
      (ops.index(opdefs.L2DistanceOpNode),
       [len(input_nodes) + 1,
        len(input_nodes) + len(existing_ops) + 1])
  ]
  program_length = 0
  search_space = create_search_space(input_nodes, ddqn, program_length, ops)
  return ops, input_nodes, ddqn, search_space, num_freeze_ops, program_length
