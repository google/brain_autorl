r"""Run distributed Rainbow on Procgen, using Impala network."""
# pylint: disable=g-long-lambda
from absl import app
from absl import flags

from brain_autorl.rl_darts.algorithms.rainbow import agent_distributed
from brain_autorl.rl_darts.algorithms.rainbow import environments
from brain_autorl.rl_darts.algorithms.rainbow import nets
from brain_autorl.rl_darts.policies import base_policies
from brain_autorl.rl_darts.procgen import procgen_wrappers

import launchpad as lp

flags.DEFINE_string('exp_name', 'Rainbow Procgen', 'Experiment name.')
flags.DEFINE_string('cell', None,
                    'If empty, auto cell selector will choose cells.')
flags.DEFINE_string(
    'env_id', None, 'A valid environement id for the given environment kind.' +
    'Used in Vizier tuning. Ignored if we do hyper sweep.')
flags.DEFINE_integer('num_seeds', 1, 'Number seeds')
flags.DEFINE_integer('num_actors', 4, 'Number of actors.')
flags.DEFINE_integer('max_actor_steps', 25_000_000,
                     'Program will stop after this number is reached.')
flags.DEFINE_bool('infinite_mode', False,
                  'If true, use infinite levels for training.')
flags.DEFINE_enum('gpu_type', 'P100', ['P100', 'V100'], 'GPU type for learner.')

FLAGS = flags.FLAGS


def network_factory(spec):
  torso = base_policies.make_impala_cnn_network(
      depths=[64] * 5, use_batch_norm=False)
  return nets.C51Network(
      torso, spec.num_values, v_min=0.0, v_max=1.0, dueling_head=True)


def get_program(
    sub_env_name: str,
    seed: int = 0,
    lr: float = 5e-5,
    ise: float = 0.2,
    pe: float = 0.8,
    replay_size: int = 500_000,
    spi: float = 4,
    n_step: int = 7,
    tup: int = 500,
) -> lp.Program:
  """Build the LaunchPad program."""
  num_levels = 0 if FLAGS.infinite_mode else 200
  make_env = lambda x: environments.make_environment(
      env_name='procgen',
      sub_env_name=sub_env_name,
      seed=seed,
      evaluation=x,
      num_levels=num_levels)

  program_builder = agent_distributed.DistributedDQN(
      environment_factory=make_env,
      network_factory=network_factory,
      num_actors=FLAGS.num_actors,
      samples_per_insert=spi,
      target_update_period=tup,
      min_replay_size=10000,
      max_replay_size=replay_size,
      n_step=n_step,
      importance_sampling_exponent=ise,
      priority_exponent=pe,
      learning_rate=lr,
      max_actor_steps=FLAGS.max_actor_steps)

  return program_builder.build()


def main(_):
  if FLAGS.env_id is None:
    envs = procgen_wrappers.ENV_NAMES
  else:
    envs = [FLAGS.env_id]
  programs = [get_program(sub_env_name=env) for env in envs]

  lp.launch(programs,)


if __name__ == '__main__':
  app.run(main)
