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
