"""Reusable numerical helpers for FACTMx."""

from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf


def categorical_kl_from_logits(q_logits: Any, p_logits: Any) -> tf.Tensor:
  """Return KL[Categorical(q_logits) || Categorical(p_logits)] per item."""
  q_logits = tf.convert_to_tensor(q_logits, dtype=tf.float32)
  p_logits = tf.convert_to_tensor(p_logits, dtype=tf.float32)
  q_probs = tf.nn.softmax(q_logits, axis=-1)
  log_q = tf.nn.log_softmax(q_logits, axis=-1)
  log_p = tf.nn.log_softmax(p_logits, axis=-1)
  return tf.reduce_sum(q_probs * (log_q - log_p), axis=-1)


def np_logsumexp(values: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
  """Small NumPy log-sum-exp helper to avoid a SciPy dependency."""
  values = np.asarray(values, dtype=float)
  max_values = np.max(values, axis=axis, keepdims=True)
  stable = max_values + np.log(np.sum(np.exp(values - max_values), axis=axis, keepdims=True))
  if keepdims:
    return stable
  return np.squeeze(stable, axis=axis)
