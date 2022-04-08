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

"""Functions for creating the computational graph search space."""
import itertools
from typing import List, Tuple

from brain_autorl.evolving_rl.ops import DummyOpNode
from brain_autorl.evolving_rl.ops import LossOpNode
import numpy as np
import pyglove as pg


def product_input_ops(inputs, existing_ops, num_intermediate, operators,
                      freeze_ops):
  """"Get all possible combinations of op_idx and input node idxs to that op."""
  lst: List[Tuple[int, List[int]]] = []
  # Create dummpy ops for intermediate nodes and blank for existing ops.
  all_inputs = inputs.copy()
  if freeze_ops:
    for op_idx, input_idxs in existing_ops:
      op_cls = operators[op_idx]
      input_dtypes = [all_inputs[idx].odtype for idx in input_idxs]
      all_inputs.append(op_cls(input_dtypes=input_dtypes))

  all_inputs.extend([DummyOpNode() for _ in range(num_intermediate)])

  for op_idx, op in enumerate(operators):
    if op == LossOpNode:
      continue
    # Input_idxs is tuple of possible input nodes. Check if inputs are valid.
    iter_func = itertools.combinations
    if op.input_order_matters:
      iter_func = itertools.permutations
    for input_idxs in iter_func(range(len(all_inputs)), op.num_inputs):
      input_ops = [all_inputs[idx] for idx in input_idxs]
      if op.precheck_valid_input(
          input_ops):  # and op.check_exclusive_ops(input_ops):
        # Convert x to idx format.
        lst.append((op_idx, list(input_idxs)))
  return lst


@pg.members([
    ('program_lst',
     pg.typing.List(
         pg.typing.Tuple([pg.typing.Int(),
                          pg.typing.List(pg.typing.Int())])),
     """Each element is a (Int, List[Int]) tuple where the first int is the idx
     of the operation from the list of operators. The list of integers are the
     indexes of inputs into that operation. Indexing will be into the full list
     of inputs + ops so the first input to the program has an idx of 0.
     in operators. The second list of integers is the indexes of the operation
     inputs."""),
    ('duplicate', pg.typing.Bool(False),
     'If program is duplicate. Then skip training and just report reward'),
    ('reward', pg.typing.Float(0., min_value=-100., max_value=100.),
     """Reward is the sum of the normalized training performance across
     environments so each environment will contribute between [0, 1]. This will
     support up to 100 training environments currently and can be increased."""
    ),
    ('loss_weight', pg.typing.Float(1., min_value=0.,
                                    max_value=1.), 'Weight on loss function'),
    ('hash_output', pg.typing.Float(0.),
     """Save the hash output of a program so it does not have to be computed
     again. This is for duplicate detection.""")
])
class ProgramSpec(pg.Object):
  """Specifies a program with a list of operator indexes."""
  pass


def create_search_space(inputs,
                        existing_ops,
                        program_length,
                        operators,
                        adjust_loss_weight=False,
                        freeze_ops=True):
  """Build the search for for all DAGs."""
  program_lst = pg.List([])
  if existing_ops and freeze_ops:
    program_lst.extend(existing_ops)
  search_program_length = program_length
  if not freeze_ops:
    search_program_length = program_length + len(existing_ops)
  for i in range(search_program_length):
    program_lst.append(
        pg.oneof(
            product_input_ops(inputs, existing_ops, i, operators, freeze_ops)))
  # Assume last operator is LossOpNode
  last_op = operators[-1]
  program_lst.append(
      (operators.index(last_op), [len(inputs) + len(program_lst) - 1]))
  loss_weight = pg.oneof(last_op.loss_weights) if adjust_loss_weight else 1.0
  program_search_space = ProgramSpec(
      program_lst=program_lst,
      reward=pg.floatv(-100., 100.),
      duplicate=pg.oneof([True, False]),
      loss_weight=loss_weight,
      hash_output=pg.floatv(
          float(np.finfo(np.float32).min), float(np.finfo(np.float32).max)),
  )
  return program_search_space
