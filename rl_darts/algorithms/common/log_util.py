"""Utilities for logging."""
import csv
import os
from typing import List, TypeVar

from absl import logging
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
