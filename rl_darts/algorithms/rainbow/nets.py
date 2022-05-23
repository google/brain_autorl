"""C51 network.

The C51 + dueling setup is used in the Rainbow paper:
https://arxiv.org/abs/1710.02298
"""
from typing import Tuple
import numpy as np
import sonnet.v2 as snt
import tensorflow as tf


class C51DuelingHead(snt.Module):
  """Dueling head for C51."""

  def __init__(self,
               num_actions: int,
               num_atoms: int,
               hidden_sizes: Tuple[int, ...],
               with_noisy_linear=False,
               name='c51_dueling_head'):
    super().__init__(name)
    self._num_actions = num_actions
    self._num_atoms = num_atoms
    self._num_values = num_actions * num_atoms

    self._value_mlp = snt.nets.MLP(
        output_sizes=hidden_sizes + (self._num_atoms,), name='value_mlp')
    self._advantage_mlp = snt.nets.MLP(
        output_sizes=hidden_sizes + (self._num_values,), name='advantage_mlp')

  def __call__(self, inputs):
    inputs = snt.flatten(inputs)
    # Compute value & advantage for duelling. The values are interpreted as
    # logits (over atoms).
    value = self._value_mlp(inputs)  # [B, Atoms]
    value = tf.reshape(value, [-1, 1, self._num_atoms])

    advantages = self._advantage_mlp(inputs)  # [B, A * Atoms]
    advantages = tf.reshape(advantages,
                            [-1, self._num_actions, self._num_atoms])
    advantages -= tf.reduce_mean(advantages, axis=-2, keepdims=True)
    logits = value + advantages  # [B, A, Atoms]
    # Reshape, to be consistent with the regular MLP head.
    return tf.reshape(logits, [-1, self._num_values])


class C51Network(snt.Module):
  """C51 network."""

  def __init__(self,
               torso: snt.Module,
               num_actions: int,
               num_atoms: int = 51,
               v_min: float = -10.0,
               v_max: float = 10.0,
               dueling_head: bool = False,
               name: str = 'c51_network'):
    super().__init__(name)
    self._num_actions = num_actions
    self._num_atoms = num_atoms
    self._v_min = v_min
    self._v_max = v_max

    if dueling_head:
      head = C51DuelingHead(num_actions, num_atoms, (512,))
    else:
      head = snt.nets.MLP([512, self._num_actions * self._num_atoms])

    self._network = snt.Sequential([torso, head])

  @snt.once
  def _create_atoms(self):
    initial_atoms = np.linspace(self._v_min, self._v_max, self._num_atoms)
    self._atoms = tf.Variable(initial_atoms, dtype=tf.float32, trainable=False)

  def __call__(self, observations, **kwargs):
    self._create_atoms()
    # **kwargs passed to the first layer (i.e. the torso) in snt.Sequential.
    q_logits = self._network(observations, **kwargs)
    q_logits = tf.reshape(q_logits, (-1, self._num_actions, self._num_atoms))
    q_dist = tf.nn.softmax(q_logits)
    # Sum out the atoms dimension.
    q_values = tf.stop_gradient(tf.reduce_sum(q_dist * self._atoms, 2))
    return q_values, q_logits, self._atoms

  @property
  def torso(self):
    return self._network._layers[0]  # pylint: disable=protected-access
