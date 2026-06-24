from __future__ import annotations

from typing import Any, Mapping, Optional

import tensorflow as tf

from FACTMx.head.FACTMx_head import FACTMx_head, TensorLike


class DistanceContrast(FACTMx_head):
  head_type = 'DistanceContrast'

  def __init__(
      self,
      dim: int,
      dim_latent: int,
      head_name: str,
      distance_metric: str = 'euclidean',
      pos_cutoff: Optional[float] = None,
      neg_cutoff: Optional[float] = None,
      eps = 1E-15,
      **kwargs: Any,
  ) -> None:
    dim = 0
    super().__init__(dim, dim_latent, head_name)
    self.eps = eps
    self.pos_cutoff = pos_cutoff
    self.neg_cutoff = neg_cutoff
    if distance_metric not in ['euclidean', 'cosine', 'hamming']:
      raise ValueError(f'Unknown distance metric: {distance_metric}')
    self.distance_metric = distance_metric
    self.layers = {}
    
    self.t_vars = tuple([])

  def get_distances(self, data: TensorLike) -> tf.Tensor:
    """Return pairwise distances between data points."""
    data = tf.cast(data, tf.float32)
    if self.distance_metric == 'euclidean':
      return tf.norm(data[:, None, :] - data[None, :, :], axis=-1)
    if self.distance_metric == 'cosine':
      dot_product = tf.reduce_sum(data[:, None, :] * data[None, :, :], axis=-1)
      norms = tf.norm(data, axis=-1)
      return 1 - dot_product / (norms[None, :] * norms[:, None] + self.eps)
    if self.distance_metric == 'hamming':
      seq = tf.argmax(data, axis=-1, output_type=tf.int32)
      mismatch = tf.not_equal(seq[:, None, :], seq[None, :, :])
      return tf.reduce_sum(tf.cast(mismatch, tf.float32), axis=-1)
    raise ValueError(f'Unknown distance metric: {self.distance_metric}')

  def encode(self, data: TensorLike) -> dict[str, tf.Tensor]:
    """Return positive and/or negative pairs."""
    if (self.pos_cutoff is None) and (self.neg_cutoff is None):
      return {}
    
    encoder_kwargs = {}
    distances = self.get_distances(data)
    if self.pos_cutoff is not None:
      encoder_kwargs['positive_pairs'] = distances < self.pos_cutoff
    if self.neg_cutoff is not None:
      encoder_kwargs['negative_pairs'] = distances > self.neg_cutoff
    return {'encoder_kwargs': encoder_kwargs}

  def decode(self, latent: TensorLike, data: TensorLike) -> tf.Tensor:
    """Return no decode"""
    return tf.constant(0.)

  def loss(self, data: TensorLike, latent: TensorLike, beta: float = 1) -> tf.Tensor:
    """Return no loss."""
    return tf.constant(0.)

  def get_config(self) -> dict[str, Any]:
    """Return a JSON-serializable DistanceContrast head configuration."""
    config = super().get_config()
    config.update({
        'head_type': self.head_type,
        'distance_metric': self.distance_metric,
        'pos_cutoff': self.pos_cutoff,
        'neg_cutoff': self.neg_cutoff,
    })
    return config

  @staticmethod
  def from_config(config: Mapping[str, Any]) -> 'DistanceContrast':
    """Create a DistanceContrast head from ``config``."""
    return DistanceContrast(**dict(config))
