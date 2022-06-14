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

"""End-to-end test for search space construction and evolution.

Evaluation logic is faked.
"""

import random

from absl.testing import absltest
from brain_autorl.evolving_rl import evolution_v2
import brain_autorl.evolving_rl.graph_configs as gconfig
from brain_autorl.evolving_rl.program import build_program
import pyglove as pg


def _get_tuning_algorithm(input_nodes, existing_ops, search_space,
                          operators, program_length, num_freeze_ops,
                          adjust_loss_weight):
  seed = 1
  population_size = 5
  tournament_size = 2
  mutation_probability = 0.95

  graph_spec = evolution_v2.GraphSpec(
      template=pg.template(search_space),
      input_nodes=input_nodes,
      existing_ops=existing_ops,
      program_length=program_length,
      operators=operators,
      adjust_loss_weight=adjust_loss_weight,
  )
  graph_hasher = evolution_v2.GraphHasher(input_nodes)
  graph_generator = evolution_v2.GraphGenerator(
      graph_spec, graph_hasher, seed)
  graph_mutator = evolution_v2.GraphMutator(
      graph_spec=graph_spec,
      graph_hasher=graph_hasher,
      mutation_probability=mutation_probability,
      num_freeze_ops=num_freeze_ops)

  return evolution_v2.build_regularized_evolution(
      population_size=population_size,
      tournament_size=tournament_size,
      seed=seed,
      graph_generator=graph_generator,
      graph_mutator=graph_mutator,
  )


class RunSearchTest(absltest.TestCase):

  def test_end_to_end_no_crash(self):
    (operators, input_nodes, existing_ops, search_space, num_freeze_ops,
     program_length) = gconfig.pre_graph_3()  # Use a smaller space for test.

    generator = _get_tuning_algorithm(
        input_nodes,
        existing_ops,
        search_space,
        operators,
        program_length,
        num_freeze_ops,
        adjust_loss_weight=False,
    )

    def fake_reward(program):
      del program
      return random.random()

    max_trials = 20
    for program_spec, feedback in pg.sample(
        search_space, generator, max_trials):  # pytype: disable=wrong-arg-types  # gen-stub-imports

      loss_program, _ = build_program(
          input_nodes, program_spec, operators, check_path_diff=0)

      reward = fake_reward(loss_program)
      all_metrics = {'reward': reward}
      feedback.add_measurement(
          reward=reward,
          metrics=all_metrics,
          step=0,
          elapse_secs=1.0,
      )
      feedback.done()


if __name__ == '__main__':
  absltest.main()
