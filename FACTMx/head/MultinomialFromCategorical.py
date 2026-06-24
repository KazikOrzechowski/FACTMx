"""Multinomial likelihood head for count observations."""

from __future__ import annotations

from typing import Any, Mapping

import tensorflow as tf
import tensorflow_probability as tfp

from FACTMx.head.FACTMx_head import Distribution, FACTMx_head, LayerConfigMap, MultinomialData, TensorLike


class MultinomialFromCategorical(FACTMx_head):
  """Multinomial likelihood head for count observations."""

  head_type = 'MultinomialFromCategorical'

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
          [tf.keras.Input(shape=(self.dim,)), tf.keras.layers.Dense(_dim_logits)]
      )
    else:
      self.layers['logits'] = tf.keras.Sequential.from_config(logits_config)  # type: ignore[arg-type]

    assert self.layers['logits'].output_shape == (None, _dim_logits)
    assert self.layers['logits'].input_shape == (None, self.dim)

    self.t_vars = tuple(self.layers['logits'].trainable_variables)

    preencoder_config = layer_configs.pop('preencoder', 'linear')
    if preencoder_config == 'linear':
      self.layers['preencoder'] = tf.keras.Sequential(
        [tf.keras.Input(shape=(_dim_logits,)), tf.keras.layers.Dense(self.dim)]
      )
    else:
      self.layers['preencoder'] = tf.keras.Sequential.from_config(preencoder_config)  # type: ignore[arg-type]
    self.t_vars += tuple(self.layers['preencoder'].trainable_variables)

    assert self.layers['preencoder'].output_shape == (None, self.dim)
    assert self.layers['preencoder'].input_shape == (None, _dim_logits)

  def decode_params(self, preencoded: TensorLike) -> tf.Tensor:
    """Decode logits from a preencoded representation."""
    logits = self.layers['logits'](preencoded)
    log_eps = tf.fill(tf.shape(logits), tf.math.log(self.eps))
    logits = tf.reduce_logsumexp(tf.stack([logits, log_eps]), axis=0)
    return tf.reshape(logits, shape=(-1, self.dim_pos, self.dim_cat))

  def make_decoder(self, preencoded: TensorLike, counts: TensorLike) -> Distribution:
    """Return the Multinomial decoder distribution for ``preencoded``."""
    logits = self.decode_params(preencoded)
    return tfp.distributions.Multinomial(total_count=counts, logits=logits)

  def decode(self, latent: TensorLike, data: MultinomialData) -> tf.Tensor:
    """Sample count observations from the decoder."""
    observations, counts = data

    preencoder_input = tf.reshape(observations, shape=(-1, self.dim_pos * self.dim_cat))
    preencoded = self.layers['preencoder'](preencoder_input)

    return self.make_decoder(preencoded, counts).sample()

  def encode(self, data: MultinomialData) -> dict[str, tf.Tensor]:
    """Encode count observations into flattened encoder inputs."""
    observations, counts = data

    preencoder_input = tf.reshape(observations, shape=(-1, self.dim_pos * self.dim_cat))
    encoder_input = self.layers['preencoder'](preencoder_input)

    return {'encoder_input': encoder_input,
            'skip_connection': encoder_input}

  def loss(self, data: MultinomialData, latent: TensorLike, skip_connection: TensorLike, beta: float = 1) -> tf.Tensor:
    """Return negative log-likelihood loss for Multinomial observations."""
    observations, counts = data
    log_prob = self.make_decoder(skip_connection, counts).log_prob(observations)
    loss = -tf.reduce_sum(log_prob) / tf.cast(tf.shape(observations)[0], tf.float32)
    loss += tf.reduce_sum(self.layers['logits'].losses)
    return loss

  def get_config(self) -> dict[str, Any]:
    """Return a JSON-serializable MultinomialFromCategorical head configuration."""
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
  def from_config(config: Mapping[str, Any]) -> 'MultinomialFromCategorical':
    config.pop('head_type', None)
    config.pop('dim_preencoded', None)
    """Create a MultinomialFromCategorical head from ``config``."""
    return MultinomialFromCategorical(**dict(config))
