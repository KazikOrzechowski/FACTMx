"""Shared mixture-head machinery for FACTMx heads."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import tensorflow as tf
import tensorflow_probability as tfp

from FACTMx.head.FACTMx_head import Distribution, FACTMx_head, LayerConfigMap, TensorLike


class Mixture(FACTMx_head):
  """Base class for heads with latent-conditioned mixture assignments.

  This class builds the common ``mixture_logits`` and ``encoder_classifier``
  layers and implements the assignment distribution, mixture-logit decoding,
  encoder preprocessing, and loss used by concrete mixture heads.
  """

  def __init__(
      self,
      dim: int,
      dim_latent: int,
      head_name: str,
      classifier_input_dim: int,
      layer_configs: LayerConfigMap = None,
      temperature: float = 1E-4,
      eps: float = 1E-3,
      mixture_logits_kernel_initializer: str = 'orthogonal',
      mixture_logits_bias_initializer: str = 'ones',
      encoder_classifier_bias_initializer: Optional[str] = None,
  ) -> None:
    super().__init__(dim, dim_latent, head_name)
    self.temperature = temperature
    self.eps = eps
    self.classifier_input_dim = classifier_input_dim
    layer_configs = dict(layer_configs or {})

    mixture_logits_config = layer_configs.pop('mixture_logits', 'linear')
    if mixture_logits_config == 'linear':
      self.layers['mixture_logits'] = tf.keras.Sequential(
          [tf.keras.Input(shape=(self.dim_latent,)),
           tf.keras.layers.Dense(
               units=self.dim,
               activation='log_softmax',
               kernel_initializer=mixture_logits_kernel_initializer,
               bias_initializer=mixture_logits_bias_initializer,
           )]
      )
    else:
      self.layers['mixture_logits'] = tf.keras.Sequential.from_config(mixture_logits_config)  # type: ignore[arg-type]

    assert self.layers['mixture_logits'].output_shape == (None, self.dim)
    assert self.layers['mixture_logits'].input_shape == (None, self.dim_latent)

    encoder_classifier_config = layer_configs.pop('encoder_classifier', 'linear')
    if encoder_classifier_config == 'linear':
      dense_kwargs: dict[str, Any] = {'units': self.dim, 'activation': 'log_softmax'}
      if encoder_classifier_bias_initializer is not None:
        dense_kwargs['bias_initializer'] = encoder_classifier_bias_initializer
      self.layers['encoder_classifier'] = tf.keras.Sequential(
          [tf.keras.Input(shape=(None, self.classifier_input_dim)),
           tf.keras.layers.Dense(**dense_kwargs)]
      )
    else:
      self.layers['encoder_classifier'] = tf.keras.Sequential.from_config(encoder_classifier_config)  # type: ignore[arg-type]

    assert self.layers['encoder_classifier'].input_shape == (None, None, self.classifier_input_dim)
    assert self.layers['encoder_classifier'].output_shape == (None, None, self.dim)

    self.t_vars = self._common_trainable_variables()

  def _common_trainable_variables(self) -> list[tf.Variable]:
    """Return trainable variables for layers owned by ``Mixture``."""
    return [
        *self.layers['mixture_logits'].trainable_variables,
        *self.layers['encoder_classifier'].trainable_variables,
    ]

  def _set_trainable_variables(self, *extra_variables: tf.Variable) -> None:
    """Set ``t_vars`` from common variables plus concrete-head variables."""
    self.t_vars = [*self._common_trainable_variables(), *extra_variables]

  def get_assignment_distribution(self, logits: TensorLike) -> Distribution:
    """Return a relaxed categorical distribution over assignments."""
    return tfp.distributions.RelaxedOneHotCategorical(logits=logits, temperature=self.temperature)

  def decode_mixture_logits(self, latent: TensorLike) -> tf.Tensor:
    """Decode log mixture proportions from latent representations."""
    mixture_logits = self.layers['mixture_logits'](latent)
    log_eps = tf.constant(tf.math.log(self.eps), shape=mixture_logits.shape)
    return tf.reduce_logsumexp(tf.stack([mixture_logits, log_eps]), axis=0)

  def get_component_log_likelihoods(self, data: TensorLike) -> tf.Tensor:
    """Return per-component log likelihoods for ``data``.

    Concrete subclasses must return a tensor broadcast-compatible with
    ``(batch, subbatch, dim)``.
    """
    raise NotImplementedError

  def decode(
      self,
      latent: TensorLike,
      data: TensorLike,
      sample: bool = True,
  ) -> tuple[Optional[tf.Tensor], tf.Tensor, tf.Tensor]:
    """Decode assignment samples and logits for ``data`` conditioned on ``latent``."""
    mixture_logits = self.decode_mixture_logits(latent)
    mixture_logits = tf.reshape(mixture_logits, (-1, 1, self.dim))

    log_likelihoods = self.get_component_log_likelihoods(data)
    assignment_logits = tf.math.add(mixture_logits, log_likelihoods)
    assignment_sample = self.get_assignment_distribution(assignment_logits).sample() if sample else None
    return assignment_sample, assignment_logits, mixture_logits

  def loss(
      self,
      data: TensorLike,
      latent: TensorLike,
      encoder_assignment_sample: TensorLike,
      encoder_assignment_logits: TensorLike,
      beta: float = 1,
  ) -> tf.Tensor:
    """Return KL plus expected negative log-likelihood for a mixture head."""
    _assignment_sample, assignment_logits, mixture_logits = self.decode(latent, data, sample=False)
    log_likelihoods = tf.math.subtract(assignment_logits, mixture_logits)

    kl_divergence = tf.reduce_mean(
        tfp.distributions.OneHotCategorical(logits=encoder_assignment_logits).kl_divergence(
            tfp.distributions.OneHotCategorical(logits=mixture_logits)
        )
    )

    log_likelihood = tf.reduce_sum(tf.math.multiply(encoder_assignment_sample, log_likelihoods))
    batch_size, subbatch_size, _ = data.shape
    ll_loss = -log_likelihood / batch_size / subbatch_size

    return tf.reduce_sum([
        kl_divergence,
        ll_loss,
        *self.layers['mixture_logits'].losses,
        *self.layers['encoder_classifier'].losses,
    ])

  def encode(self, data: TensorLike) -> dict[str, tf.Tensor]:
    """Encode observations into log mixture-proportion encoder inputs."""
    assignment_logits = self.layers['encoder_classifier'](data)
    assignment_sample = self.get_assignment_distribution(assignment_logits).sample()

    proportions_sample = tf.reduce_mean(assignment_sample, axis=1) + self.eps
    encoder_input = tf.math.log(proportions_sample)

    return {
        'encoder_input': encoder_input,
        'encoder_assignment_sample': assignment_sample,
        'encoder_assignment_logits': assignment_logits,
    }

  def get_config(self) -> dict[str, Any]:
    """Return common mixture-head configuration values."""
    config = super().get_config()
    config.update({
        'temperature': self.temperature,
        'eps': self.eps,
        'layer_configs': {key: layer.get_config() for key, layer in self.layers.items()},
    })
    return config
