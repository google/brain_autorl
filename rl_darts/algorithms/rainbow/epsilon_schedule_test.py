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

"""Tests for brain_autorl.rl_darts.algorithms.rainbow.epsilon_schedule."""
from absl.testing import absltest

from brain_autorl.rl_darts.algorithms.rainbow import epsilon_schedule

import sonnet as snt
import tensorflow as tf


class FakeNetwork(snt.Module):

  def __call__(self, x):
    batch = x.shape[0]
    q_values = tf.constant([[0., 0., 1., 0.0, 0.0, 0.0]])
    return tf.tile(q_values, [batch, 1])


class EpsilonScheduleTest(absltest.TestCase):

  def test_linear_schedule(self):
    linear = epsilon_schedule.LinearSchedule(
        init_value=1.0, final_value=0.0, scheduled_steps=2)
    self.assertEqual(linear.value(step=0), 1.0)
    self.assertEqual(linear.value(step=1), 0.5)
    self.assertEqual(linear.value(step=2), 0.0)

  def test_atari_schedule(self):
    atari = epsilon_schedule.AtariEpsilonSchedule(init_value=0.5)
    self.assertEqual(atari.value(0), 0.5)
    self.assertEqual(atari.value(25e6), 0.5**8)

  def test_actor_with_epsilon_schedule(self):
    network = FakeNetwork()
    linear = epsilon_schedule.LinearSchedule(
        init_value=1.0, final_value=0.0, scheduled_steps=2)
    actor = epsilon_schedule.FeedForwardActorWithEpsilonSchedule(
        schedule=linear, counter=None, policy_network=network)
    obs = tf.constant([1., 1.])
    # These are random.
    print(actor.select_action(obs))
    print(actor.select_action(obs))
    # Now epsilon should be 0.0. Deterministically choose action 2 based on the
    # fake Q values.
    self.assertEqual(actor.select_action(obs), 2)
    self.assertEqual(actor.select_action(obs), 2)


if __name__ == '__main__':
  absltest.main()
