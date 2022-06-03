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

"""Generates specified environments wrapped correctly."""

from acme import wrappers

from brain_autorl.rl_darts.procgen import procgen_wrappers

import dm_env


def make_environment(env_name: str,
                     sub_env_name: str,
                     seed: int = 0,
                     evaluation: bool = False,
                     **env_kwargs) -> dm_env.Environment:
  """Generates environments compatible with ACME algorithms."""
  if env_name == 'procgen':
    assert procgen_wrappers, 'Procgen was not imported correctly!'
    if evaluation:
      env = procgen_wrappers.AcmeProcgenEnv(
          env_name=sub_env_name,
          seed=seed,
          distribution_mode='easy',
          num_levels=10000000,
          start_level=env_kwargs.pop('num_levels', 200),
          **env_kwargs)
    else:
      env = procgen_wrappers.AcmeProcgenEnv(
          env_name=sub_env_name,
          seed=seed,
          distribution_mode='easy',
          num_levels=env_kwargs.pop('num_levels', 200),
          start_level=0,
          **env_kwargs)
  env = wrappers.SinglePrecisionWrapper(env)
  return env
