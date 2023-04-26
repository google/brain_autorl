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

"""Modules for constructing evolutionary algorithms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import typing
from absl import logging

from brain_autorl.evolving_rl.ops import LossOpNode
from brain_autorl.evolving_rl.ops import Node
from brain_autorl.evolving_rl.program import build_program
from brain_autorl.evolving_rl.program import get_possible_input_idxs
from brain_autorl.evolving_rl.program import Program
from brain_autorl.evolving_rl.program import sample_valid_program_spec
from brain_autorl.evolving_rl.program_search import ProgramSpec
import numpy as np
import pyglove as pg

_GRAPH_HASH_KEY = 'graph_hash'


@pg.members([
    ('template', pg.typing.Object(pg.hyper.Template)),
    ('input_nodes', pg.typing.List(pg.typing.Object(Node))),
    ('existing_ops', pg.typing.List(pg.typing.Any())),
    ('program_length', pg.typing.Int()),
    ('operators', pg.typing.List(pg.typing.Any())),
    ('adjust_loss_weight', pg.typing.Bool()),
])
class GraphSpec(pg.Object):
  """Holds graph spec and related methods for program and program specs."""

  def sample_valid_program_spec(self) -> ProgramSpec:
    return sample_valid_program_spec(
        self.input_nodes,
        self.existing_ops,
        self.program_length,
        self.operators,
        self.adjust_loss_weight,
    )

  def build_program_from_spec(
      self, program_spec: ProgramSpec) -> typing.Tuple[Program, bool]:
    program, valid = build_program(
        self.input_nodes,
        program_spec,
        self.operators,
        0,
    )
    return program, valid


@pg.members([
    ('input_nodes', pg.typing.List(pg.typing.Object(Node))),
])
class GraphHasher(pg.Object):
  """Graph hasher."""

  def _on_bound(self):
    # Initialize random set of inputs to test for duplicates.
    self.random_inputs = [
        x.initialize_random_input(10) for x in self.input_nodes
    ]
    self.rand_float = np.random.uniform()

  def compute_hash_key(self, program) -> float:
    output = program(*self.random_inputs)
    hash_key = sum([round(x, 4) for x in output.numpy().ravel()])
    if np.isnan(hash_key) or np.isinf(hash_key):
      hash_key = self.rand_float
    hash_key = float(
        np.clip(hash_key,
                np.finfo(np.float32).min,
                np.finfo(np.float32).max))
    return hash_key


@pg.members([
    ('graph_spec', pg.typing.Object(GraphSpec)),
    ('graph_hasher', pg.typing.Object(GraphHasher)),
    ('mutation_probability', pg.typing.Float(default=0.8),
     'Probability that a mutation will take place.'),
    ('num_freeze_ops', pg.typing.Int(), 'Num ops to freeze'),
])
class GraphMutator(pg.evolution.Mutator):
  """A class to carry out a point mutation in the DAG search space."""

  allow_symbolic_assignment = True

  def _on_bound(self):
    super()._on_bound()
    self.template = self.graph_spec.template
    self.input_nodes = self.graph_spec.input_nodes
    self.operators = self.graph_spec.operators
    self.adjust_loss_weight = self.graph_spec.adjust_loss_weight

  def _check_duplicate(self, program, global_state):
    hash_key = self.graph_hasher.compute_hash_key(program)
    cache = global_state.get('cache', {})
    if hash_key in cache:
      return hash_key, True, cache[hash_key]
    else:
      return hash_key, False, 0.

  def mutate(self, dna: pg.DNA, global_state, step) -> pg.DNA:  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
    """Mutates a DNA by randomizing one of the operations."""
    del step

    old_spec = dna.spec
    if random.random() < self.mutation_probability:
      program_spec = self.template.decode(dna)
      program_lst = program_spec.program_lst
      loss_program, _ = self.graph_spec.build_program_from_spec(program_spec)

      while True:
        new_program_spec = self._alter_node_idx(dna, program_lst, loss_program)
        new_loss_program, valid = self.graph_spec.build_program_from_spec(
            new_program_spec)
        if not valid:
          continue
        hash_key, duplicate, reward = self._check_duplicate(
            new_loss_program, global_state)
        new_dna = self.template.encode(
            ProgramSpec(
                program_lst=new_program_spec.program_lst,
                duplicate=duplicate,
                reward=float(reward),
                loss_weight=new_program_spec.loss_weight))
        new_dna.set_userdata(_GRAPH_HASH_KEY, hash_key)
        logging.info('%s, %s', program_spec.loss_weight,
                     new_program_spec.loss_weight)
        if valid:
          new_dna.use_spec(old_spec)
          return new_dna
    else:
      # Sample new program
      program_spec = self.graph_spec.sample_valid_program_spec()
      program, _ = self.graph_spec.build_program_from_spec(program_spec)
      hash_key = self.graph_hasher.compute_hash_key(program)
      dna = self.template.encode(program_spec)
      dna.set_userdata(_GRAPH_HASH_KEY, hash_key)
      dna.use_spec(old_spec)
      return dna

  def _sample_node_idx_to_mutate(
      self,
      program_lst,
      loss_program,
      only_leaf=False,
  ):
    idxs = list(
        range(
            len(self.input_nodes) + self.num_freeze_ops,
            len(loss_program.ops_lst)))

    if not self.adjust_loss_weight:
      del idxs[-1]
    node_idx = random.choice(idxs)
    n_idx = node_idx - len(self.input_nodes)
    op_to_mutate = loss_program.ops_lst[node_idx]

    logging.info('Mutating op: %s', op_to_mutate)

    return node_idx, n_idx, op_to_mutate

  def _alter_node_idx(self, dna, program_lst, loss_program):
    node_idx, n_idx, op_to_replace = self._sample_node_idx_to_mutate(
        program_lst, loss_program)

    if isinstance(op_to_replace, LossOpNode):
      return self._alter_continuous(dna, program_lst, loss_program)

    is_leaf = node_idx in loss_program.find_leaf_nodes()

    # Get all possible inputs
    all_inputs = loss_program.ops_lst[:node_idx - 1]
    # Find all possible ops and inputs to switch at node idx
    possible_ops = list(set(self.operators) - set([LossOpNode]))
    valid_lst = []
    count = 0
    for op_cls in possible_ops:
      op_idx = self.operators.index(op_cls)
      for input_idxs in get_possible_input_idxs(op_cls, range(len(all_inputs))):
        count += 1
        input_ops = [all_inputs[idx] for idx in input_idxs]
        input_dtypes = [x.odtype for x in input_ops]
        if op_cls.precheck_valid_input(input_ops):
          # If leaf node, then output dtype can be anything.
          if is_leaf:
            valid_lst.append((op_idx, list(input_idxs)))
          # Otherwise check that output dtype is same.
          else:
            new_op = op_cls(
                input_idxs=list(input_idxs), input_dtypes=input_dtypes)
            if new_op.odtype == op_to_replace.odtype:
              valid_lst.append((op_idx, list(input_idxs)))
    op_idx, input_idxs = random.choice(valid_lst)
    new_program_lst = []
    for i, (old_idx, old_input_idxs) in enumerate(program_lst):
      if i == n_idx:
        new_program_lst.append((op_idx, input_idxs))
      else:
        new_program_lst.append((old_idx, old_input_idxs))
    return ProgramSpec(program_lst=new_program_lst)

  def _alter_continuous(self, dna, program_lst, loss_program):
    return ProgramSpec(
        program_lst=program_lst,
        loss_weight=random.choice(loss_program.ops_lst[-1].loss_weights))


@pg.members([
    ('graph_spec', pg.typing.Object(GraphSpec)),
    ('graph_hasher', pg.typing.Object(GraphHasher)),
    ('seed', pg.typing.Int().noneable()),
])
class GraphGenerator(pg.DNAGenerator):
  """Generates random graphs."""

  def _setup(self):
    """Sets up the generator."""
    if self.seed is None:
      self._random = random
    else:
      self._random = random.Random(self.seed)

  def _propose(self) -> pg.DNA:
    program_spec = self.graph_spec.sample_valid_program_spec()
    program, _ = self.graph_spec.build_program_from_spec(program_spec)
    hash_key = self.graph_hasher.compute_hash_key(program)
    dna = self.graph_spec.template.encode(program_spec)
    dna.set_userdata(_GRAPH_HASH_KEY, hash_key)
    dna.use_spec(self.dna_spec)
    return dna


def update_cache(
    dna_list: typing.List[pg.DNA],
    global_state,
) -> typing.List[pg.DNA]:
  """Updates the functional equivalence cache."""
  if 'cache' not in global_state:
    global_state['cache'] = {}
  cache = global_state['cache']
  for dna in dna_list:
    if _GRAPH_HASH_KEY not in dna.userdata:
      raise RuntimeError(
          f'{_GRAPH_HASH_KEY} not found in dna.userdata. Did you forget to set '
          'it during graph generation or mutation?')
    hash_key = dna.userdata[_GRAPH_HASH_KEY]
    if hash_key not in cache:
      hash_value = pg.evolution.get_fitness(dna)
      cache[hash_key] = hash_value

  logging.info('global cache updated: total %s items', len(cache))
  return dna_list


def build_regularized_evolution(
    population_size: int,
    tournament_size: int,
    seed: int,
    graph_generator: GraphGenerator,
    graph_mutator: GraphMutator,
) -> pg.evolution.Evolution:
  return pg.evolution.Evolution(
      reproduction=(
          # Tournament selection and mutation.
          pg.evolution.selectors.Random(tournament_size, seed=seed) >>
          pg.evolution.selectors.Top(1) >> graph_mutator),
      population_init=(graph_generator, population_size),
      population_update=(
          # Pop out oldest individual and update functional equivalence cache.
          pg.evolution.selectors.Last(population_size) >> update_cache))
