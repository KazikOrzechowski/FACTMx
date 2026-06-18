"""Multinomial likelihood head for count observations."""

from __future__ import annotations

from typing import Any, Mapping

import tensorflow as tf
import tensorflow_probability as tfp

from FACTMx.head.FACTMx_head import Distribution, FACTMx_head, LayerConfigMap, MultinomialData, TensorLike


class MultinomialFromCategoricalPrior(FACTMx_head):
  """Multinomial likelihood head for count observations."""

  head_type = 'MultinomialFromCategoricalPrior'

  def __init__(
      self,
      dim_pos: int,
      dim_cat: int,
      dim: int,
      dim_latent: int,
      head_name: str,
      layer_configs: LayerConfigMap = None,
      prior_locs = None, prior_scales = None,
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


    preencoder_config = layer_configs.pop('preencoder', 'linear')
    if preencoder_config == 'linear':
      self.layers['preencoder'] = tf.keras.Sequential(
        [tf.keras.Input(shape=(_dim_logits,)), tf.keras.layers.Dense(self.dim)]
      )
    else:
      self.layers['preencoder'] = tf.keras.Sequential.from_config(preencoder_config)  # type: ignore[arg-type]

    assert self.layers['preencoder'].output_shape == (None, self.dim)
    assert self.layers['preencoder'].input_shape == (None, _dim_logits)

    preencoder_config = layer_configs.pop('preencoder_scale', 'linear')
    if preencoder_config == 'linear':
      self.layers['preencoder_scale'] = tf.keras.Sequential(
        [tf.keras.Input(shape=(_dim_logits,)), tf.keras.layers.Dense(self.dim, activation='softplus')]
      )
    else:
      self.layers['preencoder_scale'] = tf.keras.Sequential.from_config(preencoder_config)  # type: ignore[arg-type]

    assert self.layers['preencoder_scale'].output_shape == (None, self.dim)
    assert self.layers['preencoder_scale'].input_shape == (None, _dim_logits)

    if prior_locs is None:
      prior_locs = tf.keras.initializers.Orthogonal(self.dim ** .5)((self.dim_latent, self.dim))
    self.prior_locs = tf.constant(prior_locs, dtype='float32')

    if prior_scales is None:
      prior_scales = tf.ones((self.dim_latent, self.dim))
    self.prior_scales = tf.constant(prior_scales, dtype='float32')

    self.t_vars = (
      *self.layers['logits'].trainable_variables,
      *self.layers['preencoder'].trainable_variables,
      *self.layers['preencoder_scale'].trainable_variables,
    )

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

  def decode(self, latent: TensorLike, data: MultinomialData, deterministic: bool = False) -> tf.Tensor:
    """Sample count observations from the decoder."""
    observations, counts = data

    preencoder_input = tf.reshape(observations, shape=(-1, self.dim_pos * self.dim_cat))
    preencoded = self.layers['preencoder'](preencoder_input)
    if not deterministic:
      preencoded_scale = self.layers['preencoder_scale'](preencoder_input) + self.eps
      preencoded = tfp.distributions.MultivariateNormalDiag(preencoded, preencoded_scale).sample()

    return self.make_decoder(preencoded, counts).sample()

  def encode(self, data: MultinomialData) -> dict[str, tf.Tensor]:
    """Encode count observations into flattened encoder inputs."""
    observations, counts = data

    preencoder_input = tf.reshape(observations, shape=(-1, self.dim_pos * self.dim_cat))
    preencoded = self.layers['preencoder'](preencoder_input)
    preencoded_scale = self.layers['preencoder_scale'](preencoder_input) + self.eps

    encoder_input = tfp.distributions.MultivariateNormalDiag(preencoded, preencoded_scale).sample()

    return {'encoder_input': encoder_input,
            'skip_connection': encoder_input,
            'preencoded': preencoded,
            'preencoded_scale': preencoded_scale}

  def loss(self,
           data: MultinomialData,
           latent: TensorLike,
           skip_connection,
           preencoded: TensorLike,
           preencoded_scale: TensorLike,
           beta: float = 1) -> tf.Tensor:
    """Return negative log-likelihood loss for Multinomial observations."""
    observations, counts = data
    log_prob = self.make_decoder(skip_connection, counts).log_prob(observations)

    loss = -tf.reduce_sum(log_prob) / observations.shape[0]
    loss += tf.reduce_sum(self.layers['logits'].losses)
    loss += tf.reduce_sum(self.layers['preencoder'].losses)
    loss += tf.reduce_sum(self.layers['preencoder_scale'].losses)

    prior_dist = tfp.distributions.MultivariateNormalDiag(
      loc=tf.expand_dims(self.prior_locs, axis=0),
      scale_diag=tf.expand_dims(self.prior_scales, axis=0),
    )

    prior_log_prob = prior_dist.log_prob(tf.expand_dims(skip_connection, axis=1))
    decoder_log_prob = tf.reduce_logsumexp(
      tf.math.log(tf.clip_by_value(latent, self.eps, 1.0)) + prior_log_prob,
      axis=-1,
    )
    encoder_log_prob = tfp.distributions.MultivariateNormalDiag(preencoded, preencoded_scale).log_prob(skip_connection)

    kl_like = tf.reduce_sum(encoder_log_prob - decoder_log_prob)
    return loss + beta * kl_like

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
        'prior_locs': self.prior_locs.numpy().tolist(),
        'prior_scales': self.prior_scales.numpy().tolist(),
        'eps': self.eps,
        'encode_logits': self.encode_logits,
    }

  @staticmethod
  def from_config(config: Mapping[str, Any]) -> 'MultinomialFromCategoricalPrior':
    config.pop('head_type', None)
    config.pop('dim_preencoded', None)
    """Create a MultinomialFromCategoricalPrior head from ``config``."""
    return MultinomialFromCategoricalPrior(**dict(config))
