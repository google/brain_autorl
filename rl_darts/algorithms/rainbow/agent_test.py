"""Tests for C51 agent."""
from absl.testing import absltest
from absl.testing import parameterized

import acme
from acme import specs
from acme.testing import fakes

from brain_autorl.rl_darts.algorithms.rainbow import agent as agent_lib
from brain_autorl.rl_darts.algorithms.rainbow import nets

import numpy as np
import sonnet.v2 as snt


class C51Test(parameterized.TestCase):

  @parameterized.parameters({'dueling': True}, {'dueling': False})
  def test_dqn(self, dueling):
    # Create a fake environment to test with.
    environment = fakes.DiscreteEnvironment(
        num_actions=5,
        num_observations=10,
        obs_dtype=np.float32,
        episode_length=10)
    spec = specs.make_environment_spec(environment)

    torso = snt.Sequential([
        snt.Flatten(),
        snt.nets.MLP([50, 50]),
    ])
    network = nets.C51Network(
        torso, spec.actions.num_values, dueling_head=dueling)
    # Construct the agent.
    agent = agent_lib.RainbowDQN(
        environment_spec=spec,
        network=network,
        batch_size=10,
        samples_per_insert=2,
        min_replay_size=10)

    # Try running the environment loop. We have no assertions here because all
    # we care about is that the agent runs without raising any errors.
    loop = acme.EnvironmentLoop(environment, agent)
    loop.run(num_episodes=2)


if __name__ == '__main__':
  absltest.main()
