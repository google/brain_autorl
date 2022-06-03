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

from absl.testing import parameterized

from brain_autorl.rl_darts.algorithms.rainbow import nets

import sonnet.v2 as snt
import tensorflow as tf


class NetsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters({'dueling_head': False}, {'dueling_head': True})
  def test_c51_network(self, dueling_head):
    torso = snt.nets.MLP([10, 20])
    model = nets.C51Network(
        torso, num_actions=4, v_min=-1., v_max=1., dueling_head=dueling_head)
    q_values, q_logits, atoms = model(tf.ones((8, 32)))
    self.assertEqual(q_values.shape, (8, 4))
    self.assertEqual(q_logits.shape, (8, 4, 51))
    self.assertEqual(atoms.shape, (51,))


if __name__ == '__main__':
  tf.test.main()
