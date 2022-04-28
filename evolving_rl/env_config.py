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

"""Contains the configuration settings for each environment."""
import collections
import copy
from bsuite import sweep

classical_env_config = collections.OrderedDict([
    (
        'CartPole-v0',
        {
            'min_return': 0.,
            'max_return': 200.,
            'num_episodes': 400,
            'eval_episodes': 50,
            'num_steps': 2e4,
            'epsilon_schedule': 1e3,
        }),
    (
        'Acrobot-v1',
        {
            'min_return': -500.,
            'max_return': -50.,
            'num_episodes': 400,
            'eval_episodes': 50,
            'num_steps': 1e5,
            'epsilon_schedule': 1e3,
        }),
    (
        'LunarLander-v2',
        {
            'min_return': -500.,
            'max_return': 250.,
            'num_episodes': 1000,
            'eval_episodes': 50,
            'num_steps': 1e6,  # during eval
            'train_steps': 6e5,  # during meta training
        }),
    ('MountainCar-v0', {
        'min_return': -200.,
        'max_return': -110.,
        'num_episodes': 500,
        'eval_episodes': 50,
        'num_steps': 1e6
    }),
])

minigrid_env_config = dict([
    (
        'MiniGrid-Empty-5x5-v0',
        {
            'min_return': 0.,
            'max_return': 1.,
            'num_episodes': 400,
            'eval_episodes': 50,
            'num_steps': 4e5
        }),
    ('MiniGrid-Empty-6x6-v0', {
        'min_return': 0.,
        'max_return': 1.,
        'num_episodes': 400,
        'eval_episodes': 50,
        'num_steps': 1e6
    }),
    ('MiniGrid-Empty-Random-5x5-v0', {
        'min_return': 0.,
        'max_return': 1.,
        'num_episodes': 400,
        'eval_episodes': 50,
        'num_steps': 1e6
    }),
    ('MiniGrid-Dynamic-Obstacles-Random-5x5-v0', {
        'min_return': -1.,
        'max_return': 1.,
        'num_episodes': 400,
        'eval_episodes': 50,
        'num_steps': 1e6
    }),
    (
        'MiniGrid-Dynamic-Obstacles-5x5-v0',
        {
            'min_return': -1.,
            'max_return': 1.,
            'num_episodes': 400,
            'eval_episodes': 50,
            'num_steps': 1e6,
            'train_steps': 2e5,  # during meta training
        }),
    (
        'MiniGrid-Dynamic-Obstacles-6x6-v0',
        {
            'min_return': -1.,
            'max_return': 1.,
            'num_episodes': 400,
            'eval_episodes': 50,
            'num_steps': 1e6,
            'train_steps': 5e5,  # during meta training
        }),
    (
        'MiniGrid-Dynamic-Obstacles-8x8-v0',
        {
            'min_return': -1.,
            'max_return': 1.,
            'num_episodes': 400,
            'eval_episodes': 50,
            'num_steps': 1e6,
            'train_steps': 5e5,  # during meta training
        }),
    ('MiniGrid-Empty-Random-5x5-v0', {
        'min_return': 0.,
        'max_return': 1.,
        'num_episodes': 400,
        'eval_episodes': 50,
        'num_steps': 1e6
    }),
    (
        'MiniGrid-FourRooms-v0',
        {
            'min_return': 0.,
            'max_return': 1.,
            'num_episodes': 400,
            'eval_episodes': 50,
            'num_steps': 2e6,
            'train_steps': 1e6,  # during meta training
            'epsilon_schedule': 500000
        }),
    (
        'MiniGrid-Unlock-v0',
        {
            'min_return': 0.,
            'max_return': 1.,
            'num_episodes': 400,
            'eval_episodes': 50,
            'num_steps': 2e6,
            'train_steps': 5e5,  # during meta training
        }),
    (
        'MiniGrid-UnlockPickup-v0',
        {
            'min_return': 0.,
            'max_return': 1.,
            'num_episodes': 400,
            'eval_episodes': 50,
            'num_steps': 1e6,
            'train_steps': 5e5,  # during meta training
        }),
    (
        'MiniGrid-SimpleCrossingS9N1-v0',
        {
            'min_return': 0.,
            'max_return': 1.,
            'num_episodes': 400,
            'eval_episodes': 50,
            'num_steps': 2e6,
            'train_steps': 4e5,  # during meta training
        }),
    ('MiniGrid-LavaCrossingS9N1-v0', {
        'min_return': 0.,
        'max_return': 1.,
        'num_episodes': 400,
        'eval_episodes': 50,
        'num_steps': 1e6
    }),
    ('MiniGrid-LavaGapS5-v0', {
        'min_return': 0.,
        'max_return': 1.,
        'num_episodes': 400,
        'eval_episodes': 50,
        'num_steps': 1e6
    }),
    (
        'MiniGrid-DoorKey-5x5-v0',
        {
            'min_return': 0.,
            'max_return': 1.,
            'num_episodes': 400,
            'eval_episodes': 50,
            'num_steps': 1e6,
            'train_steps': 5e5,  # during meta training
        }),
    (
        'MiniGrid-DoorKey-6x6-v0',
        {
            'min_return': 0.,
            'max_return': 1.,
            'num_episodes': 400,
            'eval_episodes': 50,
            'num_steps': 1e6,
            'train_steps': 5e5,  # during meta training
        }),
    (
        'MiniGrid-KeyCorridorS3R1-v0',
        {
            'min_return': 0.,
            'max_return': 1.,
            'num_episodes': 400,
            'eval_episodes': 50,
            'num_steps': 1e6,
            'train_steps': 5e5,  # during meta training
        }),
    (
        'MiniGrid-KeyCorridorS3R2-v0',
        {
            'min_return': 0.,
            'max_return': 1.,
            'num_episodes': 400,
            'eval_episodes': 50,
            'num_steps': 2e6,
            'train_steps': 5e5,  # during meta training
            'epsilon_schedule': 500000,
        }),
    (
        'MiniGrid-MultiRoom-N2-S4-v0',
        {
            'min_return': 0.,
            'max_return': 1.,
            'num_episodes': 400,
            'eval_episodes': 50,
            'num_steps': 2e6,
            'train_steps': 1e6,  # during meta training
            'epsilon_schedule': 500000,
        }),
    (
        'MiniGrid-MultiRoom-N4-S5-v0',
        {
            'min_return': 0.,
            'max_return': 1.,
            'num_episodes': 400,
            'eval_episodes': 50,
            'num_steps': 2e6,
            'train_steps': 5e5,  # during meta training
            'epsilon_schedule': 500000,
        }),
])

minigrid_modifiers = [('ScaleR10', 10), ('ScaleR100', 100),
                      ('ScaleR1000', 1000), ('ScaleR10000', 10000)]
update_dict = {}
for k, v in minigrid_env_config.items():
  for modifier, r_scale in minigrid_modifiers:
    new_key = modifier + k
    update_dict[new_key] = copy.deepcopy(v)
    update_dict[new_key]['min_return'] *= r_scale
    update_dict[new_key]['max_return'] *= r_scale
minigrid_env_config.update(update_dict)

procgen_env_config = collections.OrderedDict([
    ('procgen:procgen-plunder-v0', {
        'min_return': 0.,
        'max_return': 250.,
        'num_steps': 25e6,
        'num_episodes': 25000,
        'eval_episodes': 50,
    }),
])

atari_env_config = collections.OrderedDict([
    ('BeamRiderNoFrameskip-v4', {
        'min_return': 0.,
        'max_return': 5000.,
    }),
    (
        'GopherNoFrameskip-v4',
        {
            'min_return': 0.,
            'max_return': 5000.,
        }),
    (
        'BoxingNoFrameskip-v4',
        {
            'min_return': 0.,
            'max_return': 5000.,
        }),
    (
        'BowlingNoFrameskip-v4',
        {
            'min_return': 0.,
            'max_return': 5000.,
        }),
    (
        'VideoPinballNoFrameskip-v4',
        {
            'min_return': 0.,
            'max_return': 5000.,
        }),
    ('AtlantisNoFrameskip-v4', {
        'min_return': 0.,
        'max_return': 5000.,
    }),
    (
        'RoadRunnerNoFrameskip-v4',
        {
            'min_return': 0.,
            'max_return': 5000.,
        }),
    (
        'BankHeistNoFrameskip-v4',
        {
            'min_return': 0.,
            'max_return': 5000.,
        }),
    (
        'SeaquestNoFrameskip-v4',
        {
            'min_return': 0.,
            'max_return': 5000.,
        }),
    (
        'BreakoutNoFrameskip-v4',
        {
            'min_return': 0.,
            'max_return': 5000.,
        }),
    (
        'QbertNoFrameskip-v4',
        {
            'min_return': 0.,
            'max_return': 5000.,
        }),
    (
        'AsteroidsNoFrameskip-v4',
        {
            'min_return': 0.,
            'max_return': 5000.,
        }),
    (
        'SkiingNoFrameskip-v4',
        {
            'min_return': 0.,
            'max_return': 5000.,
        }),
])

atari_common = {
    'num_steps': 50e6,
    'train_steps': 50e6,
    'num_episodes': 20000,
    'eval_episodes': 50,
    'eval_period': 1e6,
    'target_update_period': 1000,
    'epsilon_schedule': 1e6,
    'epsilon_final_p': 0.1,
    'learning_rate': 0.0001,
    'min_replay_size': 50000,
    'samples_per_insert': 8,  # bs / this number to get 4 steps per update
    'optimizer': 'Adam',
    'noop': True,
    'checkpoint': True,
}

bsuite_config = {x: {} for x in sweep.SWEEP}

bsuite_cartpole = {
    'min_return': 0,
    'max_return': 1000,
    'info_key': 'raw_return',
}

bsuite_cartpole_swingup = {
    'min_return': -100,
    'max_return': 1000,
    'info_key': 'raw_return',
    'num_episodes': 500,
}

bsuite_deepsea = {
    'min_return': 0,
    'max_return': 1,
    'num_episodes': 2000,
    'info_key': 'denoised_return'
}

bsuite_umbrella = {
    'min_return': -1,
    'max_return': 1,
    'num_episodes': 2000,
    'info_key': 'total_regret'
}

bsuite_common = {
    'min_return': 0,
    'max_return': 0,
    'eval_episodes': 1,
    'num_episodes': 400,
    'learning_rate': 0.0001,
    'epsilon_schedule': 1000,
    'epsilon_final_p': 0.02,
}

for k, v in bsuite_config.items():
  bsuite_config[k] = {**v, **bsuite_common}
  if 'cartpole' in k and 'swingup' not in k:
    bsuite_config[k].update(bsuite_cartpole)
  elif 'swingup' in k:
    bsuite_config[k].update(bsuite_cartpole_swingup)
  elif 'deep_sea' in k:
    bsuite_config[k].update(bsuite_deepsea)
  elif 'umbrella' in k:
    bsuite_config[k].update(bsuite_umbrella)

for k, v in atari_env_config.items():
  atari_env_config[k] = {**v, **atari_common}

full_env_config = collections.OrderedDict(
    list(classical_env_config.items()) + list(minigrid_env_config.items()) +
    list(procgen_env_config.items()) + list(atari_env_config.items()) +
    list(bsuite_config.items()))
