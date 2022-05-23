"""Tests for base_policies."""
from acme import specs
from acme import wrappers as acme_wrappers
from acme.tf import networks

from brain_autorl.rl_darts.policies import base_policies
from brain_autorl.rl_darts.procgen import procgen_wrappers

import numpy as np
import sonnet.v2 as snt
import tensorflow as tf


class BasePoliciesTest(tf.test.TestCase):

  def test_original_impala(self):
    env = procgen_wrappers.AcmeProcgenEnv(env_name='coinrun', seed=0)
    env = acme_wrappers.SinglePrecisionWrapper(env)
    environment_spec = specs.make_environment_spec(env)

    feature_network = base_policies.make_impala_cnn_network()

    original_impala = snt.Sequential([
        feature_network,
        networks.DuellingMLP(
            environment_spec.actions.num_values, hidden_sizes=[512])
    ])

    example_input = tf.zeros((1, 64, 64, 16))
    output = original_impala(example_input)
    self.assertShapeEqual(
        np.zeros((1, environment_spec.actions.num_values)), output)

  def test_custom_mlp(self):
    batch_size = 3
    mlp_size = 32
    example_image = tf.zeros((batch_size, 64, 64, 16))
    custom_mlp = base_policies.CustomMLP(
        input_inner_rank=3, output_sizes=[mlp_size])
    output = custom_mlp(example_image)
    self.assertEqual(output.shape, (batch_size, mlp_size))


if __name__ == '__main__':
  tf.test.main()
