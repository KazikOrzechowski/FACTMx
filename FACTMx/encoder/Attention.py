"""Attention-based encoder implementation for FACTMx."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp

from FACTMx.encoder.FACTMx_encoder import Distribution, FACTMx_encoder, TensorLike, all_equal


class Attention(FACTMx_encoder):
  """Attention-based encoder for equal-dimensional heads."""

  encoder_type = 'Attention'

  def __init__(
      self,
      dim_latent: int,
      head_dims: Sequence[int],
      name: Optional[str] = None,
      prior_params: Optional[Mapping[str, Any]] = None,
      eps: float = 1E-5,
  ) -> None:
    super().__init__(dim_latent, head_dims, name)
    self.eps = eps
    self.layers: dict[str, keras.Model] = {}

    assert dim_latent == self.head_dims[0]
    assert all_equal(self.head_dims)

    self.attention_mechanism = tf.keras.layers.Attention()
    self.layers['key_transform'] = keras.Sequential(
        [tf.keras.Input(shape=(dim_latent,)),
         tf.keras.layers.Dense(units=dim_latent, use_bias=False)]
    )

    self.t_vars = tuple(var for layer in self.layers.values() for var in layer.trainable_variables)

    if prior_params is None:
      loc = tf.zeros(dim_latent)
      scale_tril = tf.eye(dim_latent)
      self.prior = tfp.distributions.MultivariateNormalTriL(loc, scale_tril)
    else:
      self.prior = tfp.distributions.MultivariateNormalTriL(**prior_params)

  def encode_params(self, data: TensorLike) -> tuple[tf.Tensor, tf.Tensor]:
    """Return attention-pooled latent posterior parameters."""
    n_heads = len(self.head_dims)
    flat_data = tf.reshape(data, shape=(-1, self.dim_latent))
    keys = self.layers['key_transform'](flat_data)
    keys = tf.reshape(keys, shape=(-1, n_heads, self.dim_latent))

    broad_data = tf.reshape(data, shape=(-1, n_heads, self.dim_latent))
    values = self.attention_mechanism([keys, broad_data])

    loc = tf.reduce_mean(values, axis=1)
    scale_diag = tf.math.reduce_variance(values, axis=1) + self.eps
    scale_tril = tf.linalg.diag(scale_diag)
    return loc, scale_tril

  def make_encoder(self, data: TensorLike) -> Distribution:
    """Build the variational posterior distribution for ``data``."""
    loc, scale_tril = self.encode_params(data)
    return tfp.distributions.MultivariateNormalTriL(loc, scale_tril)

  def encode(self, data: TensorLike) -> tf.Tensor:
    """Encode observations into latent samples."""
    return self.make_encoder(data).sample()

  def encode_with_loss(self, data: TensorLike, encoder_kwargs=None) -> tuple[tf.Tensor, tf.Tensor]:
    """Encode ``data`` and return the sampled latent tensor and KL loss."""
    encoder = self.make_encoder(data)
    sample = encoder.sample()
    loss = tf.reduce_mean(encoder.kl_divergence(self.prior))
    for layer in self.layers.values():
      loss += tf.reduce_sum(layer.losses)
    return sample, loss

  def loss(self, data: TensorLike, encoder_kwargs=None) -> tf.Tensor:
    """Return KL divergence loss between posteriors and the prior."""
    encoder = self.make_encoder(data)
    loss = tf.reduce_mean(encoder.kl_divergence(self.prior))
    for layer in self.layers.values():
      loss += tf.reduce_sum(layer.losses)
    return loss

  def get_config(self) -> dict[str, Any]:
    """Return a JSON-serializable encoder configuration."""
    return {
        'encoder_type': self.encoder_type,
        'dim_latent': self.dim_latent,
        'head_dims': tuple(self.head_dims),
        'eps': self.eps,
        'prior_params': {
            'loc': self.prior.loc.numpy().tolist(),
            'scale_tril': self.prior.scale_tril.numpy().tolist(),
        },
    }

  @staticmethod
  def from_config(config: Mapping[str, Any]) -> 'Attention':
    """Create an attention encoder from ``config``."""
    return Attention(**dict(config))
