# pylint:disable=protected-access,function-redefined,unused-variable
"""Modified training file from original PI-SAC codebase."""
import os
import time
from absl import logging

from brain_autorl.rl_darts.algorithms.sac import encoders

import gin
import numpy as np
from pisac import ceb_task
from pisac import dm_control_env
from pisac import metric_utils as pisac_metric_utils
from pisac import sac_agent
from pisac import schedule_utils
from pisac import utils

from qj_global import qj

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tf_agents.agents.ddpg import critic_network
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils as tfa_metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import random_tf_policy
from tf_agents.policies import tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common
from tf_agents.utils import composite

tfd = tfp.distributions


@gin.configurable
def env_load_fn(domain_name,
                task_name,
                render_configs,
                frame_stack,
                action_repeat,
                actions_in_obs=False,
                rewards_in_obs=False):
  """Load environment."""
  env = dm_control_env.load(
      domain_name=domain_name,
      task_name=task_name,
      action_repeat=action_repeat,
      frame_stack=frame_stack,
      actions_in_obs=actions_in_obs,
      rewards_in_obs=rewards_in_obs,
      pixels_obs=True,
      render_kwargs=render_configs)
  return env


env_features = {
    ('cartpole', 'swingup'):
        dict(
            action_repeat=4, initial_collect_steps=1000,
            num_env_steps=int(2e5)),
    ('cartpole', 'balance_sparse'):
        dict(
            action_repeat=2, initial_collect_steps=1000,
            num_env_steps=int(2e5)),
    ('reacher', 'easy'):
        dict(
            action_repeat=4, initial_collect_steps=1000,
            num_env_steps=int(3e5)),
    ('ball_in_cup', 'catch'):
        dict(
            action_repeat=4, initial_collect_steps=1000,
            num_env_steps=int(2e5)),
    ('finger', 'spin'):
        dict(
            action_repeat=1,
            initial_collect_steps=10000,
            num_env_steps=int(2e5)),
    ('cheetah', 'run'):
        dict(
            action_repeat=4,
            initial_collect_steps=10000,
            num_env_steps=int(2e6)),
    ('walker', 'walk'):
        dict(
            action_repeat=2,
            initial_collect_steps=10000,
            num_env_steps=int(1e6)),
    ('walker', 'stand'):
        dict(
            action_repeat=2,
            initial_collect_steps=10000,
            num_env_steps=int(2e5)),
    ('hopper', 'stand'):
        dict(
            action_repeat=2,
            initial_collect_steps=10000,
            num_env_steps=int(1e6)),
}


@gin.configurable
def train_eval(
    root_dir,
    random_seed=None,
    # Dataset params
    domain_task_name=('cartpole', 'swingup'),
    frame_shape=(84, 84, 3),
    image_aug_type='random_shifting',  # None/'random_shifting'
    frame_stack=3,
    action_repeat=4,
    # Params for learning
    num_env_steps=1000000,
    train_darts=True,
    softmax_temperature=1.0,
    learn_ceb=True,
    use_augmented_q=False,
    # Params for CEB
    e_ctor=encoders.FRNConv,
    e_head_ctor=encoders.MVNormalDiagParamHead,
    b_ctor=encoders.FRNConv,
    b_head_ctor=encoders.MVNormalDiagParamHead,
    conv_feature_dim=50,  # deterministic feature used by actor/critic/ceb
    ceb_feature_dim=50,
    ceb_action_condition=True,
    ceb_backward_encode_rewards=True,
    initial_feature_step=0,
    feature_lr=3e-4,
    feature_lr_schedule=None,
    ceb_beta=0.01,
    ceb_beta_schedule=None,
    ceb_generative_ratio=0.0,
    ceb_generative_items=None,
    feature_grad_clip=None,
    enc_ema_tau=0.05,  # if enc_ema_tau=None, ceb also learns backend encoder
    use_critic_grad=True,
    # Params for SAC
    actor_kernel_init='glorot_uniform',
    normal_proj_net=sac_agent.sac_normal_projection_net,
    critic_kernel_init='glorot_uniform',
    critic_last_kernel_init='glorot_uniform',
    actor_fc_layers=(256, 256),
    critic_obs_fc_layers=None,
    critic_action_fc_layers=None,
    critic_joint_fc_layers=(256, 256),
    # Params for collect
    collect_every=1,
    initial_collect_steps=1000,
    collect_steps_per_iteration=1,
    replay_buffer_capacity=100000,
    # Params for target update
    target_update_tau=0.005,
    target_update_period=1,
    # Params for train
    batch_size=256,
    actor_learning_rate=3e-4,
    actor_lr_schedule=None,
    critic_learning_rate=3e-4,
    critic_lr_schedule=None,
    alpha_learning_rate=3e-4,
    alpha_lr_schedule=None,
    td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
    gamma=0.99,
    reward_scale_factor=1.0,
    gradient_clipping=None,
    use_tf_functions=True,
    drivers_in_graph=True,
    # Params for eval
    num_eval_episodes=10,
    eval_env_interval=5000,  # number of env steps
    greedy_eval_policy=True,
    train_next_frame_decoder=False,
    # Params for summaries and logging
    baseline_log_fn=None,
    checkpoint_env_interval=100000,  # number of env steps
    log_env_interval=1000,  # number of env steps
    summary_interval=1000,
    image_summary_interval=0,
    summaries_flush_secs=10,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    eval_metrics_callback=None,
    darts_metrics_callback=None,
):
  """train and eval for PI-SAC."""
  if random_seed is not None:
    tf.compat.v1.set_random_seed(random_seed)
    np.random.seed(random_seed)

  # Load env specific hparams
  domain_name, task_name = domain_task_name
  env_f = env_features[(domain_name, task_name)]
  action_repeat, initial_collect_steps, num_env_steps = (
      env_f['action_repeat'], env_f['initial_collect_steps'],
      env_f['num_env_steps'])

  # Load baseline logs and write to tensorboard
  if baseline_log_fn is not None:
    baseline_log_fn(root_dir, domain_name, task_name, action_repeat)

  if root_dir is None:
    raise AttributeError('train_eval requires a root_dir.')

  # Set iterations and intervals to be computed relative to the number of
  # environment steps rather than the number of gradient steps.
  num_iterations = (
      num_env_steps * collect_every // collect_steps_per_iteration +
      (initial_feature_step))
  checkpoint_interval = (
      checkpoint_env_interval * collect_every // collect_steps_per_iteration)
  eval_interval = (
      eval_env_interval * collect_every // collect_steps_per_iteration)
  log_interval = (
      log_env_interval * collect_every // collect_steps_per_iteration)
  logging.info('num_env_steps = %d (env steps)', num_env_steps)
  logging.info('initial_feature_step = %d (gradient steps)',
               initial_feature_step)
  logging.info('num_iterations = %d (gradient steps)', num_iterations)
  logging.info('checkpoint interval (env steps) = %d', checkpoint_env_interval)
  logging.info('checkpoint interval (gradient steps) = %d', checkpoint_interval)
  logging.info('eval interval (env steps) = %d', eval_env_interval)
  logging.info('eval interval (gradient steps) = %d', eval_interval)
  logging.info('log interval (env steps) = %d', log_env_interval)
  logging.info('log interval (gradient steps) = %d', log_interval)

  root_dir = os.path.expanduser(root_dir)

  summary_writer = tf.compat.v2.summary.create_file_writer(
      root_dir, flush_millis=summaries_flush_secs * 1000)
  summary_writer.set_as_default()

  eval_histograms = [
      pisac_metric_utils.ReturnHistogram(buffer_size=num_eval_episodes),
  ]

  eval_metrics = [
      tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
      pisac_metric_utils.ReturnStddevMetric(buffer_size=num_eval_episodes),
      tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
  ]

  # create training environment
  render_configs = {
      'height': frame_shape[0],
      'width': frame_shape[1],
      'camera_id': dict(quadruped=2).get(domain_name, 0),
  }

  tf_env = tf_py_environment.TFPyEnvironment(
      env_load_fn(domain_name, task_name, render_configs, frame_stack,
                  action_repeat))
  eval_tf_env = tf_py_environment.TFPyEnvironment(
      env_load_fn(domain_name, task_name, render_configs, frame_stack,
                  action_repeat))

  # Define global step
  g_step = common.create_variable('g_step')

  # Spec
  ims_shape = frame_shape[:2] + (frame_shape[2] * frame_stack,)
  ims_spec = tf.TensorSpec(shape=ims_shape, dtype=tf.uint8)
  conv_feature_spec = tf.TensorSpec(shape=(conv_feature_dim,), dtype=tf.float32)
  action_spec = tf_env.action_spec()

  # Forward encoder
  if train_darts:
    e_ctor = encoders.FRNDARTSConv
    b_ctor = encoders.FRNDARTSConv
  e_enc = e_ctor(
      ims_spec,
      output_dim=conv_feature_dim,
      name='e',
      softmax_temperature=softmax_temperature)
  e_enc_t = e_ctor(
      ims_spec,
      output_dim=conv_feature_dim,
      name='e_t',
      softmax_temperature=softmax_temperature)
  e_enc.create_variables()
  e_enc_t.create_variables()
  common.soft_variables_update(
      e_enc.variables, e_enc_t.variables, tau=1.0, tau_non_trainable=1.0)

  # Forward encoder head
  if e_head_ctor is None:
    e_head = None
  else:
    stacked_action_spec = tensor_spec.BoundedTensorSpec(
        action_spec.shape[:-1] + (action_spec.shape[-1] * frame_stack),
        action_spec.dtype,
        action_spec.minimum.tolist() * frame_stack,
        action_spec.maximum.tolist() * frame_stack, action_spec.name)
    e_head_spec = [conv_feature_spec, stacked_action_spec
                  ] if ceb_action_condition else conv_feature_spec
    e_head = e_head_ctor(e_head_spec, output_dim=ceb_feature_dim, name='e_head')
    e_head.create_variables()

  # Backward encoder
  b_enc = b_ctor(
      ims_spec,
      output_dim=conv_feature_dim,
      name='b',
      softmax_temperature=softmax_temperature)
  b_enc.create_variables()

  # Backward encoder head
  if b_head_ctor is None:
    b_head = None
  else:
    stacked_reward_spec = tf.TensorSpec(shape=(frame_stack,), dtype=tf.float32)
    b_head_spec = [conv_feature_spec, stacked_reward_spec
                  ] if ceb_backward_encode_rewards else conv_feature_spec
    b_head = b_head_ctor(b_head_spec, output_dim=ceb_feature_dim, name='b_head')
    b_head.create_variables()

  # future decoder for generative formulation
  future_deconv = None
  future_reward_mlp = None
  y_decoders = None
  if ceb_generative_ratio > 0.0:
    future_deconv = utils.SimpleDeconv(
        conv_feature_spec, output_tensor_spec=ims_spec)
    future_deconv.create_variables()

    future_reward_mlp = utils.MLP(
        conv_feature_spec,
        hidden_dims=(ceb_feature_dim, ceb_feature_dim // 2, frame_stack))
    future_reward_mlp.create_variables()

    y_decoders = [future_deconv, future_reward_mlp]

  m_vars = e_enc.trainable_variables
  if enc_ema_tau is None:
    m_vars += b_enc.trainable_variables
  else:  # do not train b_enc
    common.soft_variables_update(
        e_enc.variables, b_enc.variables, tau=1.0, tau_non_trainable=1.0)

  if e_head_ctor is not None:
    m_vars += e_head.trainable_variables
  if b_head_ctor is not None:
    m_vars += b_head.trainable_variables
  if ceb_generative_ratio > 0.0:
    m_vars += future_deconv.trainable_variables
    m_vars += future_reward_mlp.trainable_variables

  feature_lr_fn = schedule_utils.get_schedule_fn(
      base=feature_lr, sched=feature_lr_schedule, step=g_step)
  m_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=feature_lr_fn)

  # CEB beta schedule, e.q. 'berp@0:1.0:1000_10000:0.3:0'
  beta_fn = schedule_utils.get_schedule_fn(
      base=ceb_beta, sched=ceb_beta_schedule, step=g_step)

  def img_pred_summary_fn(obs, pred):
    utils.replay_summary(
        'y0',
        g_step,
        reshape=True,
        frame_stack=frame_stack,
        image_summary_interval=image_summary_interval)(obs, None)
    utils.replay_summary(
        'y0_pred',
        g_step,
        reshape=True,
        frame_stack=frame_stack,
        image_summary_interval=image_summary_interval)(pred, None)
    utils.replay_summary(
        'y0_pred_diff',
        g_step,
        reshape=True,
        frame_stack=frame_stack,
        image_summary_interval=image_summary_interval)(
            ((obs - pred) / 2.0 + 0.5), None)

  ceb = ceb_task.CEB(
      beta_fn=beta_fn,
      generative_ratio=ceb_generative_ratio,
      generative_items=ceb_generative_items,
      step_counter=g_step,
      img_pred_summary_fn=img_pred_summary_fn)
  m_ceb = ceb_task.CEBTask(
      ceb,
      e_enc,
      b_enc,
      forward_head=e_head,
      backward_head=b_head,
      y_decoders=y_decoders,
      learn_backward_enc=(enc_ema_tau is None),
      action_condition=ceb_action_condition,
      backward_encode_rewards=ceb_backward_encode_rewards,
      optimizer=m_optimizer,
      grad_clip=feature_grad_clip,
      global_step=g_step)

  if train_next_frame_decoder:
    ns_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
    next_frame_deconv = utils.SimpleDeconv(
        conv_feature_spec, output_tensor_spec=ims_spec)
    next_frame_decoder = utils.PixelDecoder(
        next_frame_deconv,
        optimizer=ns_optimizer,
        step_counter=g_step,
        image_summary_interval=image_summary_interval,
        frame_stack=frame_stack)
    next_frame_deconv.create_variables()

  # Agent training
  actor_lr_fn = schedule_utils.get_schedule_fn(
      base=actor_learning_rate, sched=actor_lr_schedule, step=g_step)
  critic_lr_fn = schedule_utils.get_schedule_fn(
      base=critic_learning_rate, sched=critic_lr_schedule, step=g_step)
  alpha_lr_fn = schedule_utils.get_schedule_fn(
      base=alpha_learning_rate, sched=alpha_lr_schedule, step=g_step)

  actor_net = actor_distribution_network.ActorDistributionNetwork(
      conv_feature_spec,
      action_spec,
      kernel_initializer=actor_kernel_init,
      fc_layer_params=actor_fc_layers,
      activation_fn=tf.keras.activations.relu,
      continuous_projection_net=normal_proj_net)

  critic_net = critic_network.CriticNetwork(
      (conv_feature_spec, action_spec),
      observation_fc_layer_params=critic_obs_fc_layers,
      action_fc_layer_params=critic_action_fc_layers,
      joint_fc_layer_params=critic_joint_fc_layers,
      activation_fn=tf.nn.relu,
      kernel_initializer=critic_kernel_init,
      last_kernel_initializer=critic_last_kernel_init)

  tf_agent = sac_agent.SacAgent(
      ts.time_step_spec(observation_spec=conv_feature_spec),
      action_spec,
      actor_network=actor_net,
      critic_network=critic_net,
      actor_optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=actor_lr_fn),
      critic_optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=critic_lr_fn),
      alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=alpha_lr_fn),
      target_update_tau=target_update_tau,
      target_update_period=target_update_period,
      td_errors_loss_fn=td_errors_loss_fn,
      gamma=gamma,
      reward_scale_factor=reward_scale_factor,
      gradient_clipping=gradient_clipping,
      debug_summaries=debug_summaries,
      summarize_grads_and_vars=summarize_grads_and_vars,
      train_step_counter=g_step)
  tf_agent.initialize()

  env_steps = tf_metrics.EnvironmentSteps(prefix='Train')
  average_return = tf_metrics.AverageReturnMetric(
      prefix='Train',
      buffer_size=num_eval_episodes,
      batch_size=tf_env.batch_size)
  train_metrics = [
      tf_metrics.NumberOfEpisodes(prefix='Train'), env_steps, average_return,
      tf_metrics.AverageEpisodeLengthMetric(
          prefix='Train',
          buffer_size=num_eval_episodes,
          batch_size=tf_env.batch_size),
      tf_metrics.AverageReturnMetric(
          name='LatestReturn',
          prefix='Train',
          buffer_size=1,
          batch_size=tf_env.batch_size)
  ]

  # Collect and eval policies
  initial_collect_policy = random_tf_policy.RandomTFPolicy(
      tf_env.time_step_spec(), action_spec)

  eval_policy = tf_agent.policy
  if greedy_eval_policy:
    eval_policy = greedy_policy.GreedyPolicy(eval_policy)

  def obs_to_feature(observation):
    feature, _ = e_enc(observation['pixels'], training=False)
    return tf.stop_gradient(feature)

  eval_policy = FeaturePolicy(
      policy=eval_policy,
      time_step_spec=tf_env.time_step_spec(),
      obs_to_feature_fn=obs_to_feature)

  collect_policy = FeaturePolicy(
      policy=tf_agent.collect_policy,
      time_step_spec=tf_env.time_step_spec(),
      obs_to_feature_fn=obs_to_feature)

  # Make the replay buffer.
  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      data_spec=collect_policy.trajectory_spec,
      batch_size=1,
      max_length=replay_buffer_capacity)
  replay_observer = [replay_buffer.add_batch]

  # Checkpoints
  max_to_keep = 1
  train_checkpointer = common.Checkpointer(
      ckpt_dir=os.path.join(root_dir, 'train'),
      agent=tf_agent,
      actor_net=actor_net,
      critic_net=critic_net,
      global_step=g_step,
      metrics=tfa_metric_utils.MetricsGroup(train_metrics, 'train_metrics'),
      max_to_keep=max_to_keep)
  train_checkpointer.initialize_or_restore()

  policy_checkpointer = common.Checkpointer(
      ckpt_dir=os.path.join(root_dir, 'policy'),
      policy=eval_policy,
      global_step=g_step,
      max_to_keep=max_to_keep)
  policy_checkpointer.initialize_or_restore()

  rb_checkpointer = common.Checkpointer(
      ckpt_dir=os.path.join(root_dir, 'replay_buffer'),
      max_to_keep=1,
      replay_buffer=replay_buffer,
      global_step=g_step)
  rb_checkpointer.initialize_or_restore()

  if learn_ceb:
    d = dict()
    if future_deconv is not None:
      d.update(future_deconv=future_deconv)
    if future_reward_mlp is not None:
      d.update(future_reward_mlp=future_reward_mlp)
    model_ckpt = common.Checkpointer(
        ckpt_dir=os.path.join(root_dir, 'model'),
        forward_encoder=e_enc,
        forward_encoder_target=e_enc_t,
        forward_head=e_head,
        backward_encoder=b_enc,
        backward_head=b_head,
        global_step=g_step,
        max_to_keep=max_to_keep**d)
  else:
    model_ckpt = common.Checkpointer(
        ckpt_dir=os.path.join(root_dir, 'model'),
        forward_encoder=e_enc,
        forward_encoder_target=e_enc_t,
        global_step=g_step,
        max_to_keep=max_to_keep)
  model_ckpt.initialize_or_restore()

  if train_next_frame_decoder:
    next_frame_decoder_ckpt = common.Checkpointer(
        ckpt_dir=os.path.join(root_dir, 'next_frame_decoder'),
        next_frame_decoder=next_frame_decoder,
        next_frame_deconv=next_frame_deconv,
        global_step=g_step)
    next_frame_decoder_ckpt.initialize_or_restore()

  if use_tf_functions and not drivers_in_graph:
    collect_policy.action = common.function(collect_policy.action)

  initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
      tf_env,
      initial_collect_policy,
      observers=replay_observer + train_metrics,
      num_steps=initial_collect_steps)
  collect_driver = dynamic_step_driver.DynamicStepDriver(
      tf_env,
      collect_policy,
      observers=replay_observer + train_metrics,
      num_steps=collect_steps_per_iteration)

  if use_tf_functions and drivers_in_graph:
    initial_collect_driver.run = common.function(initial_collect_driver.run)
    collect_driver.run = common.function(collect_driver.run)

  # Collect initial replay data.
  if env_steps.result() == 0 or replay_buffer.num_frames() == 0:
    qj(initial_collect_steps,
       'Initializing replay buffer by collecting random experience',
       tic=1)
    initial_collect_driver.run()
    for train_metric in train_metrics:
      train_metric.tf_summaries(train_step=env_steps.result())
    qj(s='Done initializing replay buffer', toc=1)

  time_step = None
  policy_state = collect_policy.get_initial_state(tf_env.batch_size)

  time_acc = 0
  env_steps_before = env_steps.result().numpy()

  paddings = tf.constant([[4, 4], [4, 4], [0, 0]])

  def random_shifting(traj, meta):
    x0 = traj.observation['pixels'][0]
    x1 = traj.observation['pixels'][1]
    y0 = traj.observation['pixels'][frame_stack]
    y1 = traj.observation['pixels'][frame_stack + 1]
    x0 = tf.pad(x0, paddings, 'SYMMETRIC')
    x1 = tf.pad(x1, paddings, 'SYMMETRIC')
    y0 = tf.pad(y0, paddings, 'SYMMETRIC')
    y1 = tf.pad(y1, paddings, 'SYMMETRIC')
    x0a = tf.image.random_crop(x0, ims_shape)
    x1a = tf.image.random_crop(x1, ims_shape)
    x0 = tf.image.random_crop(x0, ims_shape)
    x1 = tf.image.random_crop(x1, ims_shape)
    y0 = tf.image.random_crop(y0, ims_shape)
    y1 = tf.image.random_crop(y1, ims_shape)
    return (traj, (x0, x1, x0a, x1a, y0, y1)), meta

  # Dataset generates trajectories with shape [B, T, ...]
  num_steps = frame_stack + 2
  with tf.device('/cpu:0'):
    if image_aug_type == 'random_shifting':
      dataset = replay_buffer.as_dataset(
          sample_batch_size=batch_size, num_steps=num_steps).unbatch().filter(
              utils.filter_invalid_transition).map(
                  random_shifting, num_parallel_calls=3).batch(batch_size).map(
                      utils.replay_summary(
                          'replay/filtered',
                          order_frame_stack=True,
                          frame_stack=frame_stack,
                          image_summary_interval=image_summary_interval,
                          has_augmentations=True))
    elif image_aug_type is None:
      dataset = replay_buffer.as_dataset(
          sample_batch_size=batch_size, num_steps=num_steps).unbatch().filter(
              utils.filter_invalid_transition).batch(batch_size).map(
                  utils.replay_summary(
                      'replay/filtered',
                      order_frame_stack=True,
                      frame_stack=frame_stack,
                      image_summary_interval=image_summary_interval,
                      has_augmentations=False))
    else:
      raise NotImplementedError
  iterator_nstep = iter(dataset)

  def model_train_step(experience):
    if image_aug_type == 'random_shifting':
      experience, cropped_frames = experience
      x0, x1, _, _, y0, y1 = cropped_frames
      r0, r1, a0, a1 = utils.split_xy(
          experience, frame_stack, rewards_n_actions_only=True)
      x0 = x0[:, None, ...]
      x1 = x1[:, None, ...]
      y0 = y0[:, None, ...]
      y1 = y1[:, None, ...]
    elif image_aug_type is None:
      x0, x1, y0, y1, r0, r1, a0, a1 = utils.split_xy(
          experience, frame_stack, rewards_n_actions_only=False)
    else:
      raise NotImplementedError

    # Flatten stacked actions
    action_shape = a0.shape.as_list()
    a0 = tf.reshape(a0, [action_shape[0], action_shape[1], -1])
    a1 = tf.reshape(a1, [action_shape[0], action_shape[1], -1])

    if image_summary_interval > 0:
      utils.replay_summary(
          'ceb/x0',
          g_step,
          reshape=True,
          frame_stack=frame_stack,
          image_summary_interval=image_summary_interval)(x0, None)
      utils.replay_summary(
          'ceb/x1',
          g_step,
          reshape=True,
          frame_stack=frame_stack,
          image_summary_interval=image_summary_interval)(x1, None)
      utils.replay_summary(
          'ceb/y0',
          g_step,
          reshape=True,
          frame_stack=frame_stack,
          image_summary_interval=image_summary_interval)(y0, None)
      utils.replay_summary(
          'ceb/y1',
          g_step,
          reshape=True,
          frame_stack=frame_stack,
          image_summary_interval=image_summary_interval)(y1, None)

    ceb_loss, feat_x0, zx0 = m_ceb.train(x0, a0, y0, y1, r0, r1, m_vars)
    if train_next_frame_decoder:
      # zx0: [B, 1, Z]
      zx0 = tf.squeeze(zx0, axis=1)
      # y0: [B, 1, H, W, Cxframe_stack]
      next_obs = tf.cast(tf.squeeze(y0, axis=1), tf.float32) / 255.0
      next_frame_decoder.train(next_obs, tf.stop_gradient(zx0))

    if enc_ema_tau is not None:
      common.soft_variables_update(
          e_enc.variables,
          b_enc.variables,
          tau=enc_ema_tau,
          tau_non_trainable=enc_ema_tau)

  def agent_train_step(experience):
    # preprocess experience
    if image_aug_type == 'random_shifting':
      experience, cropped_frames = experience
      x0, x1, x0a, x1a, y0, y1 = cropped_frames
      experience = tf.nest.map_structure(
          lambda t: composite.slice_to(t, axis=1, end=2), experience)
      time_steps, actions, next_time_steps = (
          tf_agent.experience_to_transitions(experience))  # pylint: disable=protected-access
    elif image_aug_type is None:
      experience = tf.nest.map_structure(
          lambda t: composite.slice_to(t, axis=1, end=2), experience)
      time_steps, actions, next_time_steps = (
          tf_agent.experience_to_transitions(experience))  # pylint: disable=protected-access
      x0 = time_steps.observation['pixels']
      x1 = next_time_steps.observation['pixels']
    else:
      raise NotImplementedError

    tf_agent.train_pix(
        time_steps,
        actions,
        next_time_steps,
        x0,
        x1,
        x0a=x0a if use_augmented_q else None,
        x1a=x1a if use_augmented_q else None,
        e_enc=e_enc,
        e_enc_t=e_enc_t,
        q_aug=use_augmented_q,
        use_critic_grad=use_critic_grad)

  def checkpoint(step):
    # rb_checkpointer.save(global_step=step)
    train_checkpointer.save(global_step=step)
    policy_checkpointer.save(global_step=step)
    model_ckpt.save(global_step=step)
    if train_next_frame_decoder:
      next_frame_decoder_ckpt.save(global_step=step)

  def evaluate():
    # Override outer record_if that may be out of sync with respect to the
    # env_steps.result() value used for the summay step.
    with tf.compat.v2.summary.record_if(True):
      qj(g_step.numpy(), 'Starting eval at step', tic=1)
      results = pisac_metric_utils.eager_compute(
          eval_metrics,
          eval_tf_env,
          eval_policy,
          histograms=eval_histograms,
          num_episodes=num_eval_episodes,
          train_step=env_steps.result(),
          summary_writer=summary_writer,
          summary_prefix='EvalSupernet' if train_darts else 'Eval',
          use_function=drivers_in_graph,
      )
      if eval_metrics_callback is not None:
        if train_darts:
          darts_metrics_callback(results, env_steps.result())
        else:
          eval_metrics_callback(results, env_steps.result())
      tfa_metric_utils.log_metrics(eval_metrics)
      qj(s='Finished eval', toc=1)

  def update_target():
    common.soft_variables_update(
        e_enc.variables,
        e_enc_t.variables,
        tau=tf_agent.target_update_tau,
        tau_non_trainable=tf_agent.target_update_tau)
    common.soft_variables_update(
        tf_agent._critic_network_1.variables,  # pylint: disable=protected-access
        tf_agent._target_critic_network_1.variables,  # pylint: disable=protected-access
        tau=tf_agent.target_update_tau,
        tau_non_trainable=tf_agent.target_update_tau)
    common.soft_variables_update(
        tf_agent._critic_network_2.variables,  # pylint: disable=protected-access
        tf_agent._target_critic_network_2.variables,  # pylint: disable=protected-access
        tau=tf_agent.target_update_tau,
        tau_non_trainable=tf_agent.target_update_tau)

  if use_tf_functions:
    if learn_ceb:
      m_ceb.train = common.function(m_ceb.train)
      model_train_step = common.function(model_train_step)
    agent_train_step = common.function(agent_train_step)
    tf_agent.train_pix = common.function(tf_agent.train_pix)
    update_target = common.function(update_target)
    if train_next_frame_decoder:
      next_frame_decoder.train = common.function(next_frame_decoder.train)

  if not learn_ceb and initial_feature_step > 0:
    raise ValueError('Not learning CEB but initial_feature_step > 0')

  with tf.summary.record_if(
      lambda: tf.math.equal(g_step % summary_interval, 0)):
    if learn_ceb and g_step.numpy() < initial_feature_step:
      qj(initial_feature_step, 'Pretraining CEB...', tic=1)
      for _ in range(g_step.numpy(), initial_feature_step):
        with tf.name_scope('LearningRates'):
          tf.summary.scalar(
              name='CEB learning rate', data=feature_lr_fn(), step=g_step)
        experience, _ = next(iterator_nstep)
        model_train_step(experience)
        g_step.assign_add(1)
      qj(s='Done pretraining CEB.', toc=1)

  first_step = True
  for _ in range(g_step.numpy(), num_iterations):
    g_step_val = g_step.numpy()
    start_time = time.time()

    with tf.summary.record_if(
        lambda: tf.math.equal(g_step % summary_interval, 0)):

      with tf.name_scope('LearningRates'):
        tf.summary.scalar(
            name='Actor learning rate', data=actor_lr_fn(), step=g_step)
        tf.summary.scalar(
            name='Critic learning rate', data=critic_lr_fn(), step=g_step)
        tf.summary.scalar(
            name='Alpha learning rate', data=alpha_lr_fn(), step=g_step)
        if learn_ceb:
          tf.summary.scalar(
              name='CEB learning rate', data=feature_lr_fn(), step=g_step)

      with tf.name_scope('Train'):
        tf.summary.scalar(
            name='StepsVsEnvironmentSteps',
            data=env_steps.result(),
            step=g_step)
        tf.summary.scalar(
            name='StepsVsAverageReturn',
            data=average_return.result(),
            step=g_step)

      if g_step_val % collect_every == 0:
        time_step, policy_state = collect_driver.run(
            time_step=time_step,
            policy_state=policy_state,
        )

      experience, _ = next(iterator_nstep)
      agent_train_step(experience)
      if (g_step_val -
          initial_feature_step) % tf_agent.target_update_period == 0:
        update_target()
      if learn_ceb:
        model_train_step(experience)
      time_acc += time.time() - start_time

    # Increment global step counter.
    g_step.assign_add(1)
    g_step_val = g_step.numpy()

    if (g_step_val - initial_feature_step) % log_interval == 0:
      for train_metric in train_metrics:
        train_metric.tf_summaries(train_step=env_steps.result())
      logging.info('env steps = %d, average return = %f', env_steps.result(),
                   average_return.result())
      env_steps_per_sec = (env_steps.result().numpy() -
                           env_steps_before) / time_acc
      logging.info('%.3f env steps/sec', env_steps_per_sec)
      tf.compat.v2.summary.scalar(
          name='env_steps_per_sec',
          data=env_steps_per_sec,
          step=env_steps.result())
      time_acc = 0
      env_steps_before = env_steps.result().numpy()
      # Log DARTS alpha entropy here
      if hasattr(e_enc, '_conv_sequence'):
        tf.compat.v2.summary.scalar(
            name='darts entropy',
            data=e_enc._conv_sequence.net_config.normal_cell_config.alpha
            .total_mean_entropy(),
            step=env_steps.result())

    if (g_step_val - initial_feature_step) % eval_interval == 0:
      eval_start_time = time.time()
      evaluate()
      logging.info('eval time %.3f sec', time.time() - eval_start_time)

    if (g_step_val - initial_feature_step) % checkpoint_interval == 0:
      checkpoint(g_step_val)
      # Save DARTS Arch discretization here
      summ = utils.Summ(g_step, root_dir)
      if hasattr(e_enc, '_conv_sequence'):
        normal_config = e_enc._conv_sequence.net_config.normal_cell_config.to_fixed_cell_config(
            num_pred=2)
        summ.text('darts/config', normal_config.to_json_str())
        summ.flush()

    # Write gin config to Tensorboard
    if first_step:
      summ = utils.Summ(0, root_dir)
      conf = gin.operative_config_str()
      conf = '    ' + conf.replace('\n', '\n    ')
      summ.text('gin/config', conf)
      summ.flush()
      first_step = False

  # Final checkpoint.
  checkpoint(g_step.numpy())

  # Final evaluation.
  evaluate()

  if not train_darts:
    return

  eval_prefix = 'MetricsDarts'
  eval_metrics = [
      tf_metrics.AverageReturnMetric(
          buffer_size=num_eval_episodes, prefix=eval_prefix),
      pisac_metric_utils.ReturnStddevMetric(
          buffer_size=num_eval_episodes, prefix=eval_prefix),
      tf_metrics.AverageEpisodeLengthMetric(
          buffer_size=num_eval_episodes, prefix=eval_prefix)
  ]
  # DARTS Training. Retrain from scratch.
  # Define global step
  g_step = common.create_variable('g_step_darts')

  cell_config_json = e_enc._conv_sequence.net_config.normal_cell_config.to_fixed_cell_config(
      num_pred=2).to_json_str()
  # Forward encoder
  e_ctor = encoders.FRNDARTSConv
  b_ctor = encoders.FRNDARTSConv
  e_enc = e_ctor(
      ims_spec,
      output_dim=conv_feature_dim,
      name='e',
      cell_config_json=cell_config_json)
  e_enc_t = e_ctor(
      ims_spec,
      output_dim=conv_feature_dim,
      name='e_t',
      cell_config_json=cell_config_json)
  e_enc.create_variables()
  e_enc_t.create_variables()
  common.soft_variables_update(
      e_enc.variables, e_enc_t.variables, tau=1.0, tau_non_trainable=1.0)

  # Forward encoder head
  if e_head_ctor is None:
    e_head = None
  else:
    stacked_action_spec = tensor_spec.BoundedTensorSpec(
        action_spec.shape[:-1] + (action_spec.shape[-1] * frame_stack),
        action_spec.dtype,
        action_spec.minimum.tolist() * frame_stack,
        action_spec.maximum.tolist() * frame_stack, action_spec.name)
    e_head_spec = [conv_feature_spec, stacked_action_spec
                  ] if ceb_action_condition else conv_feature_spec
    e_head = e_head_ctor(e_head_spec, output_dim=ceb_feature_dim, name='e_head')
    e_head.create_variables()

  # Backward encoder
  b_enc = b_ctor(
      ims_spec,
      output_dim=conv_feature_dim,
      name='b',
      cell_config_json=cell_config_json)
  b_enc.create_variables()

  # Backward encoder head
  if b_head_ctor is None:
    b_head = None
  else:
    stacked_reward_spec = tf.TensorSpec(shape=(frame_stack,), dtype=tf.float32)
    b_head_spec = [conv_feature_spec, stacked_reward_spec
                  ] if ceb_backward_encode_rewards else conv_feature_spec
    b_head = b_head_ctor(b_head_spec, output_dim=ceb_feature_dim, name='b_head')
    b_head.create_variables()

  # future decoder for generative formulation
  future_deconv = None
  future_reward_mlp = None
  y_decoders = None
  if ceb_generative_ratio > 0.0:
    future_deconv = utils.SimpleDeconv(
        conv_feature_spec, output_tensor_spec=ims_spec)
    future_deconv.create_variables()

    future_reward_mlp = utils.MLP(
        conv_feature_spec,
        hidden_dims=(ceb_feature_dim, ceb_feature_dim // 2, frame_stack))
    future_reward_mlp.create_variables()

    y_decoders = [future_deconv, future_reward_mlp]

  m_vars = e_enc.trainable_variables
  if enc_ema_tau is None:
    m_vars += b_enc.trainable_variables
  else:  # do not train b_enc
    common.soft_variables_update(
        e_enc.variables, b_enc.variables, tau=1.0, tau_non_trainable=1.0)

  if e_head_ctor is not None:
    m_vars += e_head.trainable_variables
  if b_head_ctor is not None:
    m_vars += b_head.trainable_variables
  if ceb_generative_ratio > 0.0:
    m_vars += future_deconv.trainable_variables
    m_vars += future_reward_mlp.trainable_variables

  feature_lr_fn = schedule_utils.get_schedule_fn(
      base=feature_lr, sched=feature_lr_schedule, step=g_step)
  m_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=feature_lr_fn)

  # CEB beta schedule, e.q. 'berp@0:1.0:1000_10000:0.3:0'
  beta_fn = schedule_utils.get_schedule_fn(
      base=ceb_beta, sched=ceb_beta_schedule, step=g_step)

  ceb = ceb_task.CEB(
      beta_fn=beta_fn,
      generative_ratio=ceb_generative_ratio,
      generative_items=ceb_generative_items,
      step_counter=g_step,
      img_pred_summary_fn=img_pred_summary_fn)
  m_ceb = ceb_task.CEBTask(
      ceb,
      e_enc,
      b_enc,
      forward_head=e_head,
      backward_head=b_head,
      y_decoders=y_decoders,
      learn_backward_enc=(enc_ema_tau is None),
      action_condition=ceb_action_condition,
      backward_encode_rewards=ceb_backward_encode_rewards,
      optimizer=m_optimizer,
      grad_clip=feature_grad_clip,
      global_step=g_step)

  if train_next_frame_decoder:
    ns_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
    next_frame_deconv = utils.SimpleDeconv(
        conv_feature_spec, output_tensor_spec=ims_spec)
    next_frame_decoder = utils.PixelDecoder(
        next_frame_deconv,
        optimizer=ns_optimizer,
        step_counter=g_step,
        image_summary_interval=image_summary_interval,
        frame_stack=frame_stack)
    next_frame_deconv.create_variables()

  # Agent training
  actor_lr_fn = schedule_utils.get_schedule_fn(
      base=actor_learning_rate, sched=actor_lr_schedule, step=g_step)
  critic_lr_fn = schedule_utils.get_schedule_fn(
      base=critic_learning_rate, sched=critic_lr_schedule, step=g_step)
  alpha_lr_fn = schedule_utils.get_schedule_fn(
      base=alpha_learning_rate, sched=alpha_lr_schedule, step=g_step)

  actor_net = actor_distribution_network.ActorDistributionNetwork(
      conv_feature_spec,
      action_spec,
      kernel_initializer=actor_kernel_init,
      fc_layer_params=actor_fc_layers,
      activation_fn=tf.keras.activations.relu,
      continuous_projection_net=normal_proj_net)

  critic_net = critic_network.CriticNetwork(
      (conv_feature_spec, action_spec),
      observation_fc_layer_params=critic_obs_fc_layers,
      action_fc_layer_params=critic_action_fc_layers,
      joint_fc_layer_params=critic_joint_fc_layers,
      activation_fn=tf.nn.relu,
      kernel_initializer=critic_kernel_init,
      last_kernel_initializer=critic_last_kernel_init)

  tf_agent = sac_agent.SacAgent(
      ts.time_step_spec(observation_spec=conv_feature_spec),
      action_spec,
      actor_network=actor_net,
      critic_network=critic_net,
      actor_optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=actor_lr_fn),
      critic_optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=critic_lr_fn),
      alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=alpha_lr_fn),
      target_update_tau=target_update_tau,
      target_update_period=target_update_period,
      td_errors_loss_fn=td_errors_loss_fn,
      gamma=gamma,
      reward_scale_factor=reward_scale_factor,
      gradient_clipping=gradient_clipping,
      debug_summaries=debug_summaries,
      summarize_grads_and_vars=summarize_grads_and_vars,
      train_step_counter=g_step)
  tf_agent.initialize()

  env_steps = tf_metrics.EnvironmentSteps(prefix='TrainDarts')
  average_return = tf_metrics.AverageReturnMetric(
      prefix='TrainDarts',
      buffer_size=num_eval_episodes,
      batch_size=tf_env.batch_size)
  train_metrics = [
      tf_metrics.NumberOfEpisodes(prefix='TrainDarts'), env_steps,
      average_return,
      tf_metrics.AverageEpisodeLengthMetric(
          prefix='TrainDarts',
          buffer_size=num_eval_episodes,
          batch_size=tf_env.batch_size),
      tf_metrics.AverageReturnMetric(
          name='LatestReturn',
          prefix='TrainDarts',
          buffer_size=1,
          batch_size=tf_env.batch_size)
  ]

  # Collect and eval policies
  initial_collect_policy = random_tf_policy.RandomTFPolicy(
      tf_env.time_step_spec(), action_spec)

  eval_policy = tf_agent.policy
  if greedy_eval_policy:
    eval_policy = greedy_policy.GreedyPolicy(eval_policy)

  def obs_to_feature(observation):
    feature, _ = e_enc(observation['pixels'], training=False)
    return tf.stop_gradient(feature)

  eval_policy = FeaturePolicy(
      policy=eval_policy,
      time_step_spec=tf_env.time_step_spec(),
      obs_to_feature_fn=obs_to_feature)

  collect_policy = FeaturePolicy(
      policy=tf_agent.collect_policy,
      time_step_spec=tf_env.time_step_spec(),
      obs_to_feature_fn=obs_to_feature)

  # Make the replay buffer.
  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      data_spec=collect_policy.trajectory_spec,
      batch_size=1,
      max_length=replay_buffer_capacity)
  replay_observer = [replay_buffer.add_batch]

  # Checkpoints
  train_checkpointer = common.Checkpointer(
      ckpt_dir=os.path.join(root_dir, 'traindarts'),
      agent=tf_agent,
      actor_net=actor_net,
      critic_net=critic_net,
      global_step=g_step,
      metrics=tfa_metric_utils.MetricsGroup(train_metrics,
                                            'traindarts_metrics'),
      max_to_keep=max_to_keep)
  train_checkpointer.initialize_or_restore()

  policy_checkpointer = common.Checkpointer(
      ckpt_dir=os.path.join(root_dir, 'policydarts'),
      policy=eval_policy,
      global_step=g_step,
      max_to_keep=max_to_keep)
  policy_checkpointer.initialize_or_restore()

  rb_checkpointer = common.Checkpointer(
      ckpt_dir=os.path.join(root_dir, 'replaydarts_buffer'),
      max_to_keep=1,
      replay_buffer=replay_buffer,
      global_step=g_step)
  rb_checkpointer.initialize_or_restore()

  if learn_ceb:
    d = dict()
    if future_deconv is not None:
      d.update(future_deconv=future_deconv)
    if future_reward_mlp is not None:
      d.update(future_reward_mlp=future_reward_mlp)
    model_ckpt = common.Checkpointer(
        ckpt_dir=os.path.join(root_dir, 'modeldarts'),
        forward_encoder=e_enc,
        forward_encoder_target=e_enc_t,
        forward_head=e_head,
        backward_encoder=b_enc,
        backward_head=b_head,
        global_step=g_step,
        max_to_keep=max_to_keep**d)
  else:
    model_ckpt = common.Checkpointer(
        ckpt_dir=os.path.join(root_dir, 'modeldarts'),
        forward_encoder=e_enc,
        forward_encoder_target=e_enc_t,
        global_step=g_step,
        max_to_keep=max_to_keep)
  model_ckpt.initialize_or_restore()

  if train_next_frame_decoder:
    next_frame_decoder_ckpt = common.Checkpointer(
        ckpt_dir=os.path.join(root_dir, 'next_frame_decoder'),
        next_frame_decoder=next_frame_decoder,
        next_frame_deconv=next_frame_deconv,
        global_step=g_step)
    next_frame_decoder_ckpt.initialize_or_restore()

  if use_tf_functions and not drivers_in_graph:
    collect_policy.action = common.function(collect_policy.action)

  initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
      tf_env,
      initial_collect_policy,
      observers=replay_observer + train_metrics,
      num_steps=initial_collect_steps)
  collect_driver = dynamic_step_driver.DynamicStepDriver(
      tf_env,
      collect_policy,
      observers=replay_observer + train_metrics,
      num_steps=collect_steps_per_iteration)

  if use_tf_functions and drivers_in_graph:
    initial_collect_driver.run = common.function(initial_collect_driver.run)
    collect_driver.run = common.function(collect_driver.run)

  # Collect initial replay data.
  if env_steps.result() == 0 or replay_buffer.num_frames() == 0:
    qj(initial_collect_steps,
       'Initializing replay buffer by collecting random experience',
       tic=1)
    initial_collect_driver.run()
    for train_metric in train_metrics:
      train_metric.tf_summaries(train_step=env_steps.result())
    qj(s='Done initializing replay buffer', toc=1)

  time_step = None
  policy_state = collect_policy.get_initial_state(tf_env.batch_size)

  time_acc = 0
  env_steps_before = env_steps.result().numpy()

  paddings = tf.constant([[4, 4], [4, 4], [0, 0]])

  # Dataset generates trajectories with shape [B, T, ...]
  num_steps = frame_stack + 2
  with tf.device('/cpu:0'):
    if image_aug_type == 'random_shifting':
      dataset = replay_buffer.as_dataset(
          sample_batch_size=batch_size, num_steps=num_steps).unbatch().filter(
              utils.filter_invalid_transition).map(
                  random_shifting, num_parallel_calls=3).batch(batch_size).map(
                      utils.replay_summary(
                          'replay/filtered',
                          order_frame_stack=True,
                          frame_stack=frame_stack,
                          image_summary_interval=image_summary_interval,
                          has_augmentations=True))
    elif image_aug_type is None:
      dataset = replay_buffer.as_dataset(
          sample_batch_size=batch_size, num_steps=num_steps).unbatch().filter(
              utils.filter_invalid_transition).batch(batch_size).map(
                  utils.replay_summary(
                      'replay/filtered',
                      order_frame_stack=True,
                      frame_stack=frame_stack,
                      image_summary_interval=image_summary_interval,
                      has_augmentations=False))
    else:
      raise NotImplementedError
  iterator_nstep = iter(dataset)

  def model_train_step(experience):
    if image_aug_type == 'random_shifting':
      experience, cropped_frames = experience
      x0, x1, _, _, y0, y1 = cropped_frames
      r0, r1, a0, a1 = utils.split_xy(
          experience, frame_stack, rewards_n_actions_only=True)
      x0 = x0[:, None, ...]
      x1 = x1[:, None, ...]
      y0 = y0[:, None, ...]
      y1 = y1[:, None, ...]
    elif image_aug_type is None:
      x0, x1, y0, y1, r0, r1, a0, a1 = utils.split_xy(
          experience, frame_stack, rewards_n_actions_only=False)
    else:
      raise NotImplementedError

    # Flatten stacked actions
    action_shape = a0.shape.as_list()
    a0 = tf.reshape(a0, [action_shape[0], action_shape[1], -1])
    a1 = tf.reshape(a1, [action_shape[0], action_shape[1], -1])

    if image_summary_interval > 0:
      utils.replay_summary(
          'ceb/x0',
          g_step,
          reshape=True,
          frame_stack=frame_stack,
          image_summary_interval=image_summary_interval)(x0, None)
      utils.replay_summary(
          'ceb/x1',
          g_step,
          reshape=True,
          frame_stack=frame_stack,
          image_summary_interval=image_summary_interval)(x1, None)
      utils.replay_summary(
          'ceb/y0',
          g_step,
          reshape=True,
          frame_stack=frame_stack,
          image_summary_interval=image_summary_interval)(y0, None)
      utils.replay_summary(
          'ceb/y1',
          g_step,
          reshape=True,
          frame_stack=frame_stack,
          image_summary_interval=image_summary_interval)(y1, None)

    ceb_loss, feat_x0, zx0 = m_ceb.train(x0, a0, y0, y1, r0, r1, m_vars)
    if train_next_frame_decoder:
      # zx0: [B, 1, Z]
      zx0 = tf.squeeze(zx0, axis=1)
      # y0: [B, 1, H, W, Cxframe_stack]
      next_obs = tf.cast(tf.squeeze(y0, axis=1), tf.float32) / 255.0
      next_frame_decoder.train(next_obs, tf.stop_gradient(zx0))

    if enc_ema_tau is not None:
      common.soft_variables_update(
          e_enc.variables,
          b_enc.variables,
          tau=enc_ema_tau,
          tau_non_trainable=enc_ema_tau)

  def agent_train_step(experience):
    # preprocess experience
    if image_aug_type == 'random_shifting':
      experience, cropped_frames = experience
      x0, x1, x0a, x1a, y0, y1 = cropped_frames
      experience = tf.nest.map_structure(
          lambda t: composite.slice_to(t, axis=1, end=2), experience)
      time_steps, actions, next_time_steps = (
          tf_agent.experience_to_transitions(experience))  # pylint: disable=protected-access
    elif image_aug_type is None:
      experience = tf.nest.map_structure(
          lambda t: composite.slice_to(t, axis=1, end=2), experience)
      time_steps, actions, next_time_steps = (
          tf_agent.experience_to_transitions(experience))  # pylint: disable=protected-access
      x0 = time_steps.observation['pixels']
      x1 = next_time_steps.observation['pixels']
    else:
      raise NotImplementedError

    tf_agent.train_pix(
        time_steps,
        actions,
        next_time_steps,
        x0,
        x1,
        x0a=x0a if use_augmented_q else None,
        x1a=x1a if use_augmented_q else None,
        e_enc=e_enc,
        e_enc_t=e_enc_t,
        q_aug=use_augmented_q,
        use_critic_grad=use_critic_grad)

  def checkpoint(step):
    # rb_checkpointer.save(global_step=step)
    train_checkpointer.save(global_step=step)
    policy_checkpointer.save(global_step=step)
    model_ckpt.save(global_step=step)
    if train_next_frame_decoder:
      next_frame_decoder_ckpt.save(global_step=step)

  def evaluate():
    # Override outer record_if that may be out of sync with respect to the
    # env_steps.result() value used for the summay step.
    with tf.compat.v2.summary.record_if(True):
      qj(g_step.numpy(), 'Starting eval at step', tic=1)
      results = pisac_metric_utils.eager_compute(
          eval_metrics,
          eval_tf_env,
          eval_policy,
          histograms=eval_histograms,
          num_episodes=num_eval_episodes,
          train_step=env_steps.result(),
          summary_writer=summary_writer,
          summary_prefix='Eval',
          use_function=drivers_in_graph,
      )
      if eval_metrics_callback is not None:
        eval_metrics_callback(results, env_steps.result())
      tfa_metric_utils.log_metrics(eval_metrics)
      qj(s='Finished eval', toc=1)

  def update_target():
    common.soft_variables_update(
        e_enc.variables,
        e_enc_t.variables,
        tau=tf_agent.target_update_tau,
        tau_non_trainable=tf_agent.target_update_tau)
    common.soft_variables_update(
        tf_agent._critic_network_1.variables,  # pylint: disable=protected-access
        tf_agent._target_critic_network_1.variables,  # pylint: disable=protected-access
        tau=tf_agent.target_update_tau,
        tau_non_trainable=tf_agent.target_update_tau)
    common.soft_variables_update(
        tf_agent._critic_network_2.variables,  # pylint: disable=protected-access
        tf_agent._target_critic_network_2.variables,  # pylint: disable=protected-access
        tau=tf_agent.target_update_tau,
        tau_non_trainable=tf_agent.target_update_tau)

  if use_tf_functions:
    if learn_ceb:
      m_ceb.train = common.function(m_ceb.train)
      model_train_step = common.function(model_train_step)
    agent_train_step = common.function(agent_train_step)
    tf_agent.train_pix = common.function(tf_agent.train_pix)
    update_target = common.function(update_target)
    if train_next_frame_decoder:
      next_frame_decoder.train = common.function(next_frame_decoder.train)

  if not learn_ceb and initial_feature_step > 0:
    raise ValueError('Not learning CEB but initial_feature_step > 0')

  with tf.summary.record_if(
      lambda: tf.math.equal(g_step % summary_interval, 0)):
    if learn_ceb and g_step.numpy() < initial_feature_step:
      qj(initial_feature_step, 'Pretraining CEB...', tic=1)
      for _ in range(g_step.numpy(), initial_feature_step):
        with tf.name_scope('LearningRates'):
          tf.summary.scalar(
              name='CEB learning rate', data=feature_lr_fn(), step=g_step)
        experience, _ = next(iterator_nstep)
        model_train_step(experience)
        g_step.assign_add(1)
      qj(s='Done pretraining CEB.', toc=1)

  first_step = True
  for _ in range(g_step.numpy(), num_iterations):
    g_step_val = g_step.numpy()
    start_time = time.time()

    with tf.summary.record_if(
        lambda: tf.math.equal(g_step % summary_interval, 0)):

      with tf.name_scope('LearningRatesDarts'):
        tf.summary.scalar(
            name='Actor learning rate', data=actor_lr_fn(), step=g_step)
        tf.summary.scalar(
            name='Critic learning rate', data=critic_lr_fn(), step=g_step)
        tf.summary.scalar(
            name='Alpha learning rate', data=alpha_lr_fn(), step=g_step)
        if learn_ceb:
          tf.summary.scalar(
              name='CEB learning rate', data=feature_lr_fn(), step=g_step)

      with tf.name_scope('TrainDarts'):
        tf.summary.scalar(
            name='StepsVsEnvironmentSteps',
            data=env_steps.result(),
            step=g_step)
        tf.summary.scalar(
            name='StepsVsAverageReturn',
            data=average_return.result(),
            step=g_step)

      if g_step_val % collect_every == 0:
        time_step, policy_state = collect_driver.run(
            time_step=time_step,
            policy_state=policy_state,
        )

      experience, _ = next(iterator_nstep)
      agent_train_step(experience)
      if (g_step_val -
          initial_feature_step) % tf_agent.target_update_period == 0:
        update_target()
      if learn_ceb:
        model_train_step(experience)
      time_acc += time.time() - start_time

    # Increment global step counter.
    g_step.assign_add(1)
    g_step_val = g_step.numpy()

    if (g_step_val - initial_feature_step) % log_interval == 0:
      for train_metric in train_metrics:
        train_metric.tf_summaries(train_step=env_steps.result())
      logging.info('env steps = %d, average return = %f', env_steps.result(),
                   average_return.result())
      env_steps_per_sec = (env_steps.result().numpy() -
                           env_steps_before) / time_acc
      logging.info('%.3f env steps/sec', env_steps_per_sec)
      tf.compat.v2.summary.scalar(
          name='env_steps_per_sec',
          data=env_steps_per_sec,
          step=env_steps.result())
      time_acc = 0
      env_steps_before = env_steps.result().numpy()

    if (g_step_val - initial_feature_step) % eval_interval == 0:
      eval_start_time = time.time()
      evaluate()
      logging.info('eval time %.3f sec', time.time() - eval_start_time)

    if (g_step_val - initial_feature_step) % checkpoint_interval == 0:
      checkpoint(g_step_val)

    # Write gin config to Tensorboard
    if first_step:
      summ = utils.Summ(0, root_dir)
      conf = gin.operative_config_str()
      conf = '    ' + conf.replace('\n', '\n    ')
      summ.text('gin/config', conf)
      summ.flush()
      first_step = False

  # Final checkpoint.
  checkpoint(g_step.numpy())

  # Final evaluation.
  evaluate()


class FeaturePolicy(tf_policy.TFPolicy):
  """Feature policy wrapper."""

  def __init__(self, policy, time_step_spec, obs_to_feature_fn=None, name=None):
    """Builds a policy wrapper."""
    super(FeaturePolicy, self).__init__(
        time_step_spec,
        policy.action_spec,
        policy.policy_state_spec,
        policy.info_spec,
        emit_log_probability=policy.emit_log_probability,
        name=name)
    self.wrapped_policy = policy
    self.obs_to_feature_fn = obs_to_feature_fn

  def _variables(self):
    variables = self.wrapped_policy.variables()
    return variables

  def _distribution(self, time_steps, policy_state):
    feature = self.obs_to_feature_fn(time_steps.observation)
    time_steps = time_steps._replace(observation=feature)
    distribution_step = self.wrapped_policy.distribution(
        time_steps, policy_state)
    return distribution_step
