"""Hyperparameter sweep file.

  These defaults are optimal, but sweeps should still be conducted over the
  "SENSITIVE" labelled hparams if possible.

"""
from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
  """Hyperparameters."""
  config = config_dict.ConfigDict()

  config.env_name = 'jumper'
  config.distribution_mode = 'easy'
  config.num_levels = 200

  # Params for conv architecture
  config.impala_depths = [16, 32, 32]  # Default used in Procgen paper.

  # Params for RNN.
  config.use_rnn = False
  config.rnn_hidden_size = 256

  # Params for collect
  config.num_parallel_actors = 64  # SENSITIVE
  config.num_iterations = 20000000
  # "nsteps" in baselines PPO notation.
  config.collect_sequence_length = 1024  # SENSITIVE for RNNs.
  config.reverb_port = None

  # Params for train
  config.minibatch_size = 2048  # SENSITIVE. Set to None if using RNNs.
  config.num_epochs = 2  # SENSITIVE
  config.learning_rate = 5e-4  # SENSITIVE

  config.entropy_regularization = 0.01
  config.importance_ratio_clipping = 0.2  # "CLIPRANGE" in PPO Baselines
  config.value_clipping = 0.2  # "CLIPRANGE" in PPO Baselines
  config.value_pred_loss_coef = 0.5
  config.gradient_clipping = 0.5

  config.normalize_observations = False
  config.use_gae = True
  config.lambda_value = 0.95
  config.use_td_lambda_return = True
  config.discount_factor = 0.999

  # Params for logging
  config.policy_save_interval = 10
  config.summary_interval = 10
  config.eval_interval = 1
  config.eval_episodes = 10
  config.debug_summaries = False
  config.summarize_grads_and_vars = False

  return config
