"""Multivariate-normal likelihood head for continuous observations."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from FACTMx.custom_keras_layers import ConstantResponse
from FACTMx.head.FACTMx_head import Distribution, FACTMx_head, LayerConfigMap, TensorLike


class MultiNormal(FACTMx_head):
  """Multivariate-normal likelihood head for continuous observations."""

  head_type = 'MultiNormal'

  def __init__(
      self,
      dim: int,
      dim_latent: int,
      head_name: str,
      layer_configs: LayerConfigMap = None,
      eps: float = 1E-3,
      **kwargs: Any,
  ) -> None:
    super().__init__(dim, dim_latent, head_name)
    self.eps = eps
    self.layers = {}
    layer_configs = dict(layer_configs or {})

    loc_config = layer_configs.pop('loc', 'linear')
    if loc_config == 'linear':
      self.layers['loc'] = tf.keras.Sequential(
          [tf.keras.Input(shape=(self.dim_latent,)),
           tf.keras.layers.Dense(units=self.dim, kernel_initializer='orthogonal')]
      )
    else:
      self.layers['loc'] = tf.keras.Sequential.from_config(loc_config)  # type: ignore[arg-type]

    scale_config = layer_configs.pop('scale', 'linear')
    if scale_config == 'linear':
      self.layers['scale'] = tf.keras.Sequential(
          [tf.keras.Input(shape=(self.dim_latent,)),
           ConstantResponse(
               units=self.dim,
               activation='exponential',
               bias_initializer={'class_name': 'Constant', 'config': {'value': np.log(eps)}},
           )]
      )
    else:
      self.layers['scale'] = tf.keras.Sequential.from_config(scale_config)  # type: ignore[arg-type]

    self.t_vars = tuple(var for layer in self.layers.values() for var in layer.trainable_variables)

  def decode_params(self, latent: TensorLike) -> tuple[tf.Tensor, tf.Tensor]:
    """Decode location and diagonal-scale tensors from ``latent``."""
    loc = self.layers['loc'](latent)
    scale_diag = self.layers['scale'](latent) + self.eps
    return loc, scale_diag

  def make_decoder(self, latent: TensorLike) -> Distribution:
    """Return the multivariate-normal decoder distribution for ``latent``."""
    loc, scale = self.decode_params(latent)
    return tfp.distributions.MultivariateNormalDiag(loc, scale)

  def encode(self, data: TensorLike) -> dict[str, tf.Tensor]:
    """Return continuous observations as encoder input."""
    return {'encoder_input': data}

  def decode(self, latent: TensorLike, data: TensorLike) -> tf.Tensor:
    """Sample continuous observations from the decoder."""
    return self.make_decoder(latent).sample()

  def loss(self, data: TensorLike, latent: TensorLike, beta: float = 1) -> tf.Tensor:
    """Return negative log-likelihood loss for continuous observations."""
    loc, scale = self.decode_params(latent)
    log_prob = tfp.distributions.MultivariateNormalDiag(loc, scale).log_prob(data)
    loss = -tf.reduce_mean(log_prob)
    for layer in self.layers.values():
      loss += tf.reduce_sum(layer.losses)
    return loss

  def get_config(self) -> dict[str, Any]:
    """Return a JSON-serializable MultiNormal head configuration."""
    config = super().get_config()
    config.update({
        'head_type': self.head_type,
        'eps': self.eps,
        'layer_configs': {key: layer.get_config() for key, layer in self.layers.items()},
    })
    return config

  @staticmethod
  def from_config(config: Mapping[str, Any]) -> 'MultiNormal':
    """Create a MultiNormal head from ``config``."""
    return MultiNormal(**dict(config))
