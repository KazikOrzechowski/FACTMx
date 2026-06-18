"""Multinomial likelihood head with stochastic categorical preencoding and clone prior."""

from __future__ import annotations

from typing import Any, Mapping

import tensorflow as tf
import tensorflow_probability as tfp

from FACTMx.head.FACTMx_head import Distribution, FACTMx_head, LayerConfigMap, MultinomialData, TensorLike


class MultinomialFromCategoricalPrior(FACTMx_head):
  """Multinomial likelihood head for count observations.

  This head reconstructs sequence/count observations from a stochastic
  preencoded sequence representation. The preencoded representation is also
  regularized toward a latent-clone-conditioned Gaussian mixture prior:

      q(s | x_sequence)  versus  sum_k q_clone(k | cell) p(s | clone=k)

  The cluster-specific Gaussian prior locations and scales are trainable.
  The contribution of this prior-alignment term is controlled by
  ``sequence_prior_weight``.
  """

  head_type = 'MultinomialFromCategoricalPrior'

  def __init__(
      self,
      dim_pos: int,
      dim_cat: int,
      dim: int,
      dim_latent: int,
      head_name: str,
      layer_configs: LayerConfigMap = None,
      prior_locs = None,
      prior_scales = None,
      sequence_prior_weight: float = 1.0,
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
    self.sequence_prior_weight = float(sequence_prior_weight)
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
      prior_locs = tf.keras.initializers.Orthogonal()(
        shape=(self.dim_latent, self.dim),
        dtype=tf.float32,
      )
      prior_locs = prior_locs * tf.sqrt(tf.cast(self.dim, tf.float32))
    prior_locs = tf.convert_to_tensor(prior_locs, dtype=tf.float32)

    if prior_locs.shape != (self.dim_latent, self.dim):
      raise ValueError(
        f"prior_locs must have shape ({self.dim_latent}, {self.dim}); "
        f"got {prior_locs.shape}."
      )

    self.prior_locs = tf.Variable(
      prior_locs,
      trainable=True,
      dtype=tf.float32,
      name=f'{self.head_name}_prior_locs',
    )

    if prior_scales is None:
      prior_scales = tf.ones((self.dim_latent, self.dim), dtype=tf.float32)
    prior_scales = tf.convert_to_tensor(prior_scales, dtype=tf.float32)

    if prior_scales.shape != (self.dim_latent, self.dim):
      raise ValueError(
        f"prior_scales must have shape ({self.dim_latent}, {self.dim}); "
        f"got {prior_scales.shape}."
      )

    # Store unconstrained scale parameters and expose positive scales through
    # get_prior_scales(). This keeps the trainable scale parameters valid.
    prior_scales = tf.maximum(prior_scales, self.eps)
    self.prior_scale_logits = tf.Variable(
      tfp.math.softplus_inverse(prior_scales - self.eps),
      trainable=True,
      dtype=tf.float32,
      name=f'{self.head_name}_prior_scale_logits',
    )

    self.t_vars = (
      tuple(self.layers['logits'].trainable_variables)
      + tuple(self.layers['preencoder'].trainable_variables)
      + tuple(self.layers['preencoder_scale'].trainable_variables)
      + (self.prior_locs, self.prior_scale_logits)
    )

  def get_prior_scales(self) -> tf.Tensor:
    """Return positive trainable prior scales with shape ``(dim_latent, dim)``."""
    return tf.nn.softplus(self.prior_scale_logits) + self.eps

  def decode_params(self, preencoded: TensorLike) -> tf.Tensor:
    """Decode logits from a preencoded representation."""
    logits = self.layers['logits'](preencoded)
    log_eps = tf.fill(tf.shape(logits), tf.math.log(tf.cast(self.eps, logits.dtype)))
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
      preencoded_scale = self.layers['preencoder_scale'](preencoder_input)
      preencoded_scale = tf.maximum(preencoded_scale, self.eps)
      preencoded = tfp.distributions.MultivariateNormalDiag(preencoded, preencoded_scale).sample()

    return self.make_decoder(preencoded, counts).sample()

  def encode(self, data: MultinomialData) -> dict[str, tf.Tensor]:
    """Encode count observations into stochastic flattened encoder inputs."""
    observations, counts = data

    preencoder_input = tf.reshape(observations, shape=(-1, self.dim_pos * self.dim_cat))
    preencoded = self.layers['preencoder'](preencoder_input)
    preencoded_scale = self.layers['preencoder_scale'](preencoder_input)
    preencoded_scale = tf.maximum(preencoded_scale, self.eps)

    encoder_input = tfp.distributions.MultivariateNormalDiag(preencoded, preencoded_scale).sample()

    return {'encoder_input': encoder_input,
            'skip_connection': encoder_input,
            'preencoded': preencoded,
            'preencoded_scale': preencoded_scale}

  def reconstruction_loss(self, data: MultinomialData, skip_connection: TensorLike) -> tf.Tensor:
    """Return the multinomial reconstruction loss only."""
    observations, counts = data
    log_prob = self.make_decoder(skip_connection, counts).log_prob(observations)
    batch_size = tf.cast(tf.shape(observations)[0], log_prob.dtype)
    return -tf.reduce_sum(log_prob) / tf.maximum(batch_size, 1.0)

  def sequence_prior_loss(
      self,
      latent: TensorLike,
      skip_connection: TensorLike,
      preencoded: TensorLike,
      preencoded_scale: TensorLike,
  ) -> tf.Tensor:
    """Return KL-like alignment loss between q(s|sequence) and p(s|clone)."""
    latent = tf.convert_to_tensor(latent, dtype=skip_connection.dtype)
    latent = tf.clip_by_value(latent, self.eps, 1.0)
    latent = latent / tf.reduce_sum(latent, axis=-1, keepdims=True)

    locs = tf.cast(tf.expand_dims(self.prior_locs, axis=0), skip_connection.dtype)
    scales = tf.cast(tf.expand_dims(self.get_prior_scales(), axis=0), skip_connection.dtype)

    prior_dist = tfp.distributions.MultivariateNormalDiag(loc=locs, scale_diag=scales)
    prior_log_prob = prior_dist.log_prob(tf.expand_dims(skip_connection, axis=1))  # batch x dim_latent
    decoder_log_prob = tf.reduce_logsumexp(tf.math.log(latent) + prior_log_prob, axis=-1)

    preencoded_scale = tf.maximum(preencoded_scale, self.eps)
    encoder_log_prob = tfp.distributions.MultivariateNormalDiag(
      loc=preencoded,
      scale_diag=preencoded_scale,
    ).log_prob(skip_connection)

    return tf.reduce_mean(encoder_log_prob - decoder_log_prob)

  def loss(self,
           data: MultinomialData,
           latent: TensorLike,
           skip_connection,
           preencoded: TensorLike,
           preencoded_scale: TensorLike,
           beta: float = 1) -> tf.Tensor:
    """Return weighted reconstruction plus sequence-prior alignment loss."""
    loss = self.reconstruction_loss(data, skip_connection)

    if self.sequence_prior_weight != 0.0:
      loss += beta * self.sequence_prior_weight * self.sequence_prior_loss(
        latent=latent,
        skip_connection=skip_connection,
        preencoded=preencoded,
        preencoded_scale=preencoded_scale,
      )

    reg_losses = []
    for layer in self.layers.values():
      reg_losses.extend(layer.losses)
    if reg_losses:
      loss += tf.add_n(reg_losses)

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
        'prior_locs': self.prior_locs.numpy().tolist(),
        'prior_scales': self.get_prior_scales().numpy().tolist(),
        'sequence_prior_weight': self.sequence_prior_weight,
        'eps': self.eps,
        'encode_logits': self.encode_logits,
    }

  @staticmethod
  def from_config(config: Mapping[str, Any]) -> 'MultinomialFromCategoricalPrior':
    """Create a MultinomialFromCategoricalPrior head from ``config``."""
    config = dict(config)
    config.pop('head_type', None)
    config.pop('dim_preencoded', None)
    return MultinomialFromCategoricalPrior(**config)
