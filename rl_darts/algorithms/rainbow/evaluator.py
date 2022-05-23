"""Custom Evaluator."""
from typing import Callable, Optional
from absl import logging

import acme

from brain_autorl.rl_darts.algorithms.common import log_util

import numpy as np

EVAL_INTERVAL = 100000
NUM_EVALS = 20

CustomLogging = Callable[[int], None]  # Arg: actor_step


class EvaluatorLoop(acme.EnvironmentLoop):
  """Environment loop for the evaluator."""

  def __init__(self,
               max_actor_steps: int,
               label: str = 'environment_loop',
               eval_id: int = 0,
               **kwargs):
    super().__init__(label=label, **kwargs)
    self._max_actor_steps = max_actor_steps
    self._logger = log_util.make_default_composite_logger(
        label=f'evaluator{eval_id}')
    self._custom_logging = None

  def run(self,
          num_episodes: Optional[int] = None,
          num_steps: Optional[int] = None):
    del num_episodes, num_steps
    next_update_step = EVAL_INTERVAL
    self._should_update = False
    while True:
      counts = self._counter.get_counts()
      actor_step = counts.get('actor_steps', 0)
      if actor_step < next_update_step:
        continue
      next_update_step += EVAL_INTERVAL
      # Force a sync then evaluate.
      self._actor.update(wait=True)
      rewards = []
      for _ in range(NUM_EVALS):
        result = self.run_episode()
        rewards.append(result['episode_return'])

      self._logger.write({
          'actor_step': actor_step,
          'mean_episode_return': np.mean(rewards)
      })
      if self._custom_logging:
        self._custom_logging(actor_step)  # pylint: disable=not-callable
      if actor_step > self._max_actor_steps:
        logging.info('Max steps reached. Shutting down work unit...')
        break
