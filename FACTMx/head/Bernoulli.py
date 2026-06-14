"""Bernoulli likelihood head for binary-valued observations."""

from __future__ import annotations

from typing import Any, Mapping

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp

from FACTMx.head.FACTMx_head import Distribution, FACTMx_head, LayerConfigMap, TensorLike


class Bernoulli(FACTMx_head):
  """Bernoulli likelihood head for binary-valued observations."""

  head_type = 'Bernoulli'

  def __init__(
      self,
      dim: int,
      dim_latent: int,
      head_name: str,
      eps: float = 1E-5,
      layer_configs: LayerConfigMap = None,
      **kwargs: Any,
  ) -> None:
    super().__init__(dim, dim_latent, head_name)
    self.eps = eps
    layer_configs = dict(layer_configs or {})

    logits_config = layer_configs.pop('logits', 'linear')
    if logits_config == 'linear':
      self.layers['logits'] = keras.Sequential(
          [keras.Input(shape=(self.dim_latent,)), keras.layers.Dense(self.dim)]
      )
    else:
      self.layers['logits'] = keras.Sequential.from_config(logits_config)  # type: ignore[arg-type]

    assert self.layers['logits'].output_shape == (None, self.dim)
    assert self.layers['logits'].input_shape == (None, self.dim_latent)

    self.t_vars = self.layers['logits'].trainable_variables

  def decode_params(self, latent: TensorLike) -> tf.Tensor:
    """Decode logits from a latent representation."""
    return self.layers['logits'](latent)

  def make_decoder(self, latent: TensorLike) -> Distribution:
    """Return the Bernoulli decoder distribution for ``latent``."""
    logits = self.decode_params(latent)
    return tfp.distributions.Bernoulli(logits=logits)

  def decode(self, latent: TensorLike, data: TensorLike) -> tf.Tensor:
    """Sample binary observations from the decoder."""
    return self.make_decoder(latent).sample()

  def encode(self, data: TensorLike) -> dict[str, tf.Tensor]:
    """Return binary observations as encoder input."""
    return {'encoder_input': data}

  def loss(self, data: TensorLike, latent: TensorLike, beta: float = 1) -> tf.Tensor:
    """Return negative log-likelihood loss for Bernoulli observations."""
    log_prob = self.make_decoder(latent).log_prob(data)
    loss = -tf.reduce_mean(log_prob)
    loss += tf.reduce_sum(self.layers['logits'].losses)
    return loss

  def get_config(self) -> dict[str, Any]:
    """Return a JSON-serializable Bernoulli head configuration."""
    config = super().get_config()
    config.update({
        'head_type': self.head_type,
        'eps': self.eps,
        'layer_configs': {'logits': self.layers['logits'].get_config()},
    })
    return config

  @staticmethod
  def from_config(config: Mapping[str, Any]) -> 'Bernoulli':
    """Create a Bernoulli head from ``config``."""
    return Bernoulli(**dict(config))
