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

"""Additional Reverb Observer required for PPO."""
import concurrent.futures

from absl import logging

import tensorflow as tf

from tf_agents.replay_buffers.reverb_utils import ReverbAddTrajectoryObserver
from tf_agents.replay_buffers.reverb_utils import ReverbTrajectorySequenceObserver

from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.utils import nest_utils


class ReverbConcurrentAddBatchObserver(object):
  """Observer that writes batched trajectories into Reverb concurrently.

  The implementation uses a thread pool executor and utilizes the fact that the
  GIL lock is released by the underlying Reverb writers.

  **Note**: This observer is designed to work with py_drivers only.
  """

  def __init__(self,
               py_client,
               table_name,
               sequence_length,
               time_step_spec,
               stride_length=1,
               allow_multi_episode_sequences=False,
               batch_size=1,
               priority=5,
               pad_end_of_episodes: bool = False,
               tile_end_of_episodes: bool = False,
               thread_pool_size=1,
               check_batch_dimension=True):
    """Creates an instance of the ReverbConcurrentAddBatchObserver.

    **Note**: This observer is designed to work with py_drivers only.

    Args:
      py_client: Python client for the reverb replay server.
      table_name: The table name(s) where samples will be written to. A string
        or a sequence of strings.
      sequence_length: The sequence_length used to write to the given table.
      time_step_spec: Time step specification of a trajectory within the batch.
        This is used to get the outer batch dimension of the batch of
        trajectories that the observer is called with.
      stride_length: The integer stride for the sliding window for overlapping
        sequences.
      allow_multi_episode_sequences: Allows sequences to go over episode
        boundaries. **NOTE**: Samples generated when data is collected with this
        flag set to True will contain episode boundaries which need to be
        handled by the user.
      batch_size: The size of trajectory batches that the observer will be
        called with.
      priority: Initial priority for new samples in the RB.
      pad_end_of_episodes: Arg passed into internal observers. See
        ReverbAddTrajectoryObserver's notes for this arg.
      tile_end_of_episodes: Arg passed into internal observers. See
        ReverbAddTrajectoryObserver's notes for this arg.
      thread_pool_size: Number of threads in the thread pool used to perform
        concurrent writes if the value is gerater than 1. No thread pool is used
        if the value is 1 (default) which is the preferred method in the case of
        small observation and action space (i.e. almost always) since in those
        particular cases the runtime overhead caused by threading is usually
        bigger than the gain from the parallel execution resulting in a net
        slowdown.
      check_batch_dimension: Defines is the dimension of the received batched
        trajectories is checked against the predefined (in constructor) batch
        size. This is crucial as the predefined batch size determines the number
        of observers opened. Having mismatch between the two may result in data
        loss without warning or error.

    Raises:
      ValueError: If table_names or sequence_lengths are not lists or their
      lengths are not equal or batch size is less than 1.
      TypeError: If non-ArraySpec or non-TensorSpec is provided in
      time_step_spec.
    """
    if thread_pool_size < 1:
      raise ValueError(
          "Only positive value is supported as thread pool size, but received: "
          "%d" % thread_pool_size)

    if batch_size < 1:
      raise ValueError("The batch size must be at least 1, but received: %d" %
                       batch_size)

    if all([
        isinstance(s, tensor_spec.TensorSpec)
        for s in tf.nest.flatten(time_step_spec)
    ]):
      self._time_step_spec = tf.nest.map_structure(tensor_spec.to_array_spec,
                                                   time_step_spec)
    elif all([
        isinstance(s, array_spec.ArraySpec)
        for s in tf.nest.flatten(time_step_spec)
    ]):
      self._time_step_spec = time_step_spec
    else:
      raise TypeError(
          "time_step_spec must contain all TensorSpec or all ArraySpec (without"
          " mixing TensorSpec and ArraySpec), but received: {}".format(
              time_step_spec))

    # Initialize the observers.
    self._batch_size = batch_size
    self._check_batch_dimension = check_batch_dimension

    def _create_observer():
      if allow_multi_episode_sequences:
        return ReverbTrajectorySequenceObserver(
            py_client,
            table_name,
            sequence_length,
            stride_length=stride_length,
            priority=priority,
            pad_end_of_episodes=pad_end_of_episodes,
            tile_end_of_episodes=tile_end_of_episodes)
      else:
        return ReverbAddTrajectoryObserver(
            py_client,
            table_name,
            sequence_length,
            stride_length=stride_length,
            priority=priority,
            pad_end_of_episodes=pad_end_of_episodes,
            tile_end_of_episodes=tile_end_of_episodes)

    self._observers = [_create_observer() for _ in range(batch_size)]
    self._batched_time_step_spec = array_spec.add_outer_dims_nest(
        self._time_step_spec, (self._batch_size,))

    # Initialize the thread pool if it is necessary.
    if thread_pool_size > 1:
      self._executor = concurrent.futures.ThreadPoolExecutor(
          max_workers=thread_pool_size)
    else:
      logging.info(
          "No threadpool is used, since the requested thread pool size is: %d",
          thread_pool_size)
      self._executor = None

  def __call__(self, batched_trajectories):
    if self._check_batch_dimension:
      # Get the batch size of data received to check against the expectation.
      outer_shape = nest_utils.get_outer_array_shape(
          batched_trajectories.step_type, self._time_step_spec.step_type)

      if len(outer_shape) != 1 or outer_shape[0] != self._batch_size:
        raise ValueError(
            "The expected batch size is {}, but the outer shape is {} which "
            "has more than 1 element or has a value different than the "
            "expected batch size.".format(self._batch_size, outer_shape[0]))

      if not array_spec.check_arrays_nest(
          batched_trajectories.step_type,
          self._batched_time_step_spec.step_type):
        raise ValueError(
            "The step type of the received trajectory is incompatible with the "
            "expected batched time step spec: {} while the received step type: "
            "{}".format(self._batched_time_step_spec.step_type,
                        batched_trajectories.step_type))

    # Start the writing calls either by using the thread pool, or sequentially.
    futures = []
    for trajectory, observer in zip(
        nest_utils.unstack_nested_arrays(batched_trajectories),
        self._observers):
      if self._executor is None:
        observer(trajectory)
      else:
        futures.append(self._executor.submit(observer, trajectory))

    # Wait for all the concurrent writes to finish if there is any.
    if self._executor is not None:
      concurrent.futures.wait(futures)

  def close(self):
    """Shuts down the underlying thread pool executor.

    It must be called after the last use of the observer. It blocks until all
    the pending futures are done executing.
    """
    for observer in self._observers:
      if self._executor is None:
        observer.close()
      else:
        self._executor.submit(lambda observer: observer.close(), observer)

    # Shut down the thread pool if it is necessary waiting for all the pending
    # tasks to finish.
    if self._executor is not None:
      self._executor.shutdown()

  def flush(self):
    """Ensures that items are pushed to the service.

    Note: The items are not always immediately pushed. This method is often
    needed when `rate_limiter_timeout_ms` is set for the replay buffer.
    By calling this method before the `learner.run()`, we ensure that there is
    enough data to be consumed.
    """
    for observer in self._observers:
      observer.flush()

  def reset(self):
    """Resets the state of the observer.

    No data observed before the reset will be pushed to the RB.
    """
    for observer in self._observers:
      observer.reset()
