"""Tests for run_ppo."""
from absl import flags
from absl.testing import parameterized

from brain_autorl.rl_darts.algorithms.ppo import run_ppo

import tensorflow as tf

FLAGS = flags.FLAGS


class RunPpoTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      {'use_rnn': False},
      # {'use_rnn': True},
  )
  def test_full_loop(self, use_rnn):
    FLAGS.config.use_rnn = use_rnn
    if use_rnn:
      FLAGS.config.minibatch_size = None
      FLAGS.config.rnn_hidden_size = 2
    else:
      FLAGS.config.minibatch_size = 128
    FLAGS.config.num_iterations = 2
    FLAGS.config.num_parallel_actors = 2
    FLAGS.config.num_epochs = 1
    FLAGS.config.impala_depths = [2]
    FLAGS.config.mlp_size = 2
    FLAGS.config.rnn_hidden_size = 3
    FLAGS.config.collect_sequence_length = 64

    run_ppo.train_eval(FLAGS.test_tmpdir)


if __name__ == '__main__':
  tf.test.main()
