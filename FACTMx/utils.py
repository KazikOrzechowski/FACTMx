"""General utility helpers for FACTMx."""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import tensorflow as tf


def scalar_config_value(value: Any) -> float:
  """Return a Python float for scalar TensorFlow, NumPy, or Python values."""
  if hasattr(value, 'numpy'):
    value = value.numpy()
  if isinstance(value, np.ndarray):
    return float(np.asarray(value).reshape(()))
  return float(value)


def select_feature(index: int) -> Callable[..., Any]:
  """Return a ``tf.data.Dataset.map`` function selecting one top-level feature."""
  def _select(*features: Any) -> Any:
    return features[index]
  return _select


def split_dataset_by_feature(dataset: tf.data.Dataset, n_features: int) -> list[tf.data.Dataset]:
  """Split a structured dataset into one dataset per top-level feature."""
  return [dataset.map(select_feature(i)) for i in range(n_features)]


def dataset_cardinality_int(dataset: tf.data.Dataset) -> Optional[int]:
  """Return finite dataset cardinality as an int, or ``None`` if unknown/infinite."""
  cardinality = dataset.cardinality().numpy()
  if cardinality < 0:
    return None
  return int(cardinality)


def shuffle_buffer_size(dataset: tf.data.Dataset, default: int = 10000) -> int:
  """Return a safe shuffle buffer size for finite or unknown-cardinality datasets."""
  cardinality = dataset_cardinality_int(dataset)
  if cardinality is None:
    return int(default)
  return max(1, int(cardinality))
