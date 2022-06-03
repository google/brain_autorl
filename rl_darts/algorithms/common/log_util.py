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

"""Utilities for logging."""
import csv
import os
from typing import List, TypeVar
from absl import logging

from acme.utils import loggers as acme_loggers
from acme.utils.loggers import tf_summary
import tensorflow as tf
from tf_agents.metrics import py_metric


def log_csv_row(csv_file, row):
  with tf.gfile.Open(csv_file, 'ab') as csvfile:
    cw = csv.writer(
        csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    cw.writerow(row)


PyMetric = TypeVar('PyMetric', bound=py_metric.PyMetric)


class MetricLogger:
  """Logger for saving PyMetrics to XM GUI and csv files."""

  def __init__(self, root_dir: str):
    self._root_dir = root_dir
    self._measurement_dict = {}

  def log_metrics(self, metrics: List[PyMetric], step: int, prefix: str):
    """Logs measurements to XM and csv files."""
    log_path = os.path.join(self._root_dir, prefix + '.csv')
    csv_column_names = ['step']
    csv_row_values = [step]

    try:
      for m in metrics:
        objective_value = m.result()
        full_metric_name = prefix + '/' + m.name
        csv_column_names.append(full_metric_name)
        csv_row_values.append(objective_value)

      if step == 0 and not tf.gfile.Exists(log_path):
        log_csv_row(log_path, csv_column_names)
      log_csv_row(log_path, csv_row_values)

    except:  # pylint:disable=bare-except
      logging.info('Failed to create measurement')


class CompositeLogger(acme_loggers.Logger):
  """Consolidate a list of loggers."""

  def __init__(self, logger_list: List[acme_loggers.Logger]):
    self._loggers = logger_list

  def write(self, data):
    for logger in self._loggers:
      logger.write(data)

  def close(self):
    for logger in self._loggers:
      logger.close()


def make_default_composite_logger(directory: str = '~/acme',
                                  label: str = '',
                                  steps_key: str = 'steps',
                                  add_csv: bool = False,
                                  tf_logdir: str = '') -> CompositeLogger:
  """A flexbile logger that can optionally add csv or tensorboard logging.

  Args:
    directory: Base directory for csv logger.
    label: Used by the default logger, csv logger, and tf logger.
    steps_key: See the docstr for `make_default_logger`. This is required by the
      `XMMeasurementLogger`. A common pitfall is: a `Counter` object with prefix
      "foo" will contain "foo_steps" in the logging data. If `steps_key` is left
      as default, the XMMeasurementLogger will complain it cannot find "steps"
      in the logging data, and will skip sending xm measurements. To fix, set
      `steps_key` to "foo_steps".
    add_csv: If true, add a CSV logger that writes to directory.
    tf_logdir: If non-empty, add a TF logger (for tensorboard).

  Returns:
    A composite logger.
  """

  logger_list = [acme_loggers.make_default_logger(label, steps_key=steps_key)]
  if add_csv:
    logger_list.append(
        acme_loggers.CSVLogger(directory_or_file=directory, label=label))
  if tf_logdir:
    logger_list.append(tf_summary.TFSummaryLogger(tf_logdir, label))
  return CompositeLogger(logger_list)
