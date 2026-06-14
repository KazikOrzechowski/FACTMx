"""Multinomial likelihood head for count observations."""

from __future__ import annotations

from typing import Any, Mapping

import tensorflow as tf
import tensorflow_probability as tfp

from FACTMx.head.FACTMx_head import Distribution, FACTMx_head, LayerConfigMap, MultinomialData, TensorLike


class Multinomial(FACTMx_head):
  """Multinomial likelihood head for count observations."""

  head_type = 'Multinomial'

  def __init__(
      self,
      dim_pos: int,
      dim_cat: int,
      dim: int,
      dim_latent: int,
      head_name: str,
      layer_configs: LayerConfigMap = None,
      eps: float = 1E-3,
      encode_logits: bool = True,
      **kwargs: Any,
  ) -> None:
    super().__init__(dim, dim_latent, head_name)
    self.eps = eps
    self.dim_pos = dim_pos
    self.dim_cat = dim_cat
    _dim_logits = dim_pos * dim_cat
    self.encode_logits = encode_logits
    layer_configs = dict(layer_configs or {})

    logits_config = layer_configs.pop('logits', 'linear')
    if logits_config == 'linear':
      self.layers['logits'] = tf.keras.Sequential(
          [tf.keras.Input(shape=(self.dim_latent,)), tf.keras.layers.Dense(_dim_logits)]
      )
    else:
      self.layers['logits'] = tf.keras.Sequential.from_config(logits_config)  # type: ignore[arg-type]

    assert self.layers['logits'].output_shape == (None, _dim_logits)
    assert self.layers['logits'].input_shape == (None, self.dim_latent)

    self.t_vars = tuple(self.layers['logits'].trainable_variables)

    preencoder_config = layer_configs.pop('preencoder', None)
    if preencoder_config is None:
      self.preencoder = False
    else:
      self.preencoder = True
      self.layers['preencoder'] = tf.keras.Sequential.from_config(preencoder_config)  # type: ignore[arg-type]
      self.t_vars += tuple(self.layers['preencoder'].trainable_variables)

      assert self.layers['preencoder'].output_shape == (None, self.dim)
      assert self.layers['preencoder'].input_shape == (None, dim_pos * dim_cat)

  def decode_params(self, latent: TensorLike) -> tf.Tensor:
    """Decode logits from a latent representation."""
    logits = self.layers['logits'](latent)
    log_eps = tf.constant(tf.math.log(self.eps), shape=logits.shape)
    logits = tf.reduce_logsumexp(tf.stack([logits, log_eps]), axis=0)
    return tf.reshape(logits, shape=(-1, self.dim_pos, self.dim_cat))

  def make_decoder(self, latent: TensorLike, counts: TensorLike) -> Distribution:
    """Return the Multinomial decoder distribution for ``latent``."""
    logits = self.decode_params(latent)
    return tfp.distributions.Multinomial(total_count=counts, logits=logits)

  def decode(self, latent: TensorLike, data: MultinomialData) -> tf.Tensor:
    """Sample count observations from the decoder."""
    _observations, counts = data
    return self.make_decoder(latent, counts).sample()

  def encode(self, data: MultinomialData) -> dict[str, tf.Tensor]:
    """Encode count observations into flattened encoder inputs."""
    observations, counts = data
    if self.preencoder:
      preencoder_input = tf.reshape(observations, shape=(-1, self.dim_pos * self.dim_cat))
      encoder_input = self.layers['preencoder'](preencoder_input)
    elif self.encode_logits:
      encoder_input = observations / tf.expand_dims(counts, -1) + self.eps
      encoder_input = tf.math.log(encoder_input)
      encoder_input = tf.reshape(encoder_input, shape=(-1, self.dim_pos * self.dim_cat))
    else:
      encoder_input = tf.reshape(observations, shape=(-1, self.dim_pos * self.dim_cat))
    return {'encoder_input': encoder_input}

  def loss(self, data: MultinomialData, latent: TensorLike, beta: float = 1) -> tf.Tensor:
    """Return negative log-likelihood loss for Multinomial observations."""
    observations, counts = data
    log_prob = self.make_decoder(latent, counts).log_prob(observations)
    loss = -tf.reduce_sum(log_prob) / observations.shape[0]
    loss += tf.reduce_sum(self.layers['logits'].losses)
    return loss

  def get_config(self) -> dict[str, Any]:
    """Return a JSON-serializable Multinomial head configuration."""
    return {
        'head_type': self.head_type,
        'dim_pos': self.dim_pos,
        'dim_cat': self.dim_cat,
        'dim': self.dim,
        'dim_latent': self.dim_latent,
        'head_name': self.head_name,
        'layer_configs': {key: layer.get_config() for key, layer in self.layers.items()},
        'eps': self.eps,
        'encode_logits': self.encode_logits,
    }

  @staticmethod
  def from_config(config: Mapping[str, Any]) -> 'Multinomial':
    """Create a Multinomial head from ``config``."""
    return Multinomial(**dict(config))
