"""Categorical/Dirichlet encoder for FACTMx.

This encoder is intended for FACTMx heads whose latent state is a discrete
assignment vector, for example a clone assignment or a topic assignment.  It
maps the concatenated head encodings to parameters of a Dirichlet variational
posterior over a probability simplex.  The posterior mean is interpreted as a
soft categorical assignment and can be used directly by simple mixture heads.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp

from FACTMx.custom_keras_layers import ConstantResponse
from FACTMx.encoder.FACTMx_encoder import Distribution, FACTMx_encoder, LayerConfigMap, TensorLike


class Categorical(FACTMx_encoder):
  """Dirichlet encoder for simplex-valued latent variables.

  The encoder builds two networks over the concatenated head encodings:

  * ``mean`` returns a probability vector of length ``dim_latent``.
  * ``confidence`` returns a positive scalar concentration multiplier.

  The variational posterior is

  ``q(z | x) = Dirichlet(mean(x) * confidence(x))``.

  Sampling from this posterior gives a relaxed categorical/probability-vector
  latent representation.  Deterministic encoding returns the posterior mean.

  Args:
    dim_latent: Number of latent categories/classes.
    head_dims: Dimensions contributed by individual heads to the shared encoder.
    layer_configs: Optional serialized Keras configs for ``mean`` and
      ``confidence`` networks.  Passing ``"linear"`` or omitting either network
      creates a small default network.
    name: Optional TensorFlow module name.
    prior_params: Optional parameters passed to ``tfp.distributions.Dirichlet``.
      If omitted, a symmetric unit Dirichlet prior is used.
    eps: Small positive floor added to posterior parameters for numerical
      stability.
  """

  encoder_type = 'Categorical'

  def __init__(
      self,
      dim_latent: int,
      head_dims: Sequence[int],
      layer_configs: LayerConfigMap = None,
      name: Optional[str] = None,
      prior_params: Optional[Mapping[str, Any]] = None,
      eps: float = 1E-5,
  ) -> None:
    super().__init__(dim_latent, head_dims, name)
    self.eps = float(eps)
    self.layers: dict[str, keras.Model] = {}
    layer_configs = dict(layer_configs or {})
    encoder_input_dim = int(sum(self.head_dims))

    mean_config = layer_configs.pop('mean', 'linear')
    if mean_config == 'linear':
      # Default posterior mean network.  The softmax activation ensures that
      # the output is a probability vector over latent categories.
      self.layers['mean'] = keras.Sequential(
          [tf.keras.Input(shape=(encoder_input_dim,)),
           tf.keras.layers.Dense(units=dim_latent, activation='softmax')]
      )
    else:
      self.layers['mean'] = keras.Sequential.from_config(mean_config)  # type: ignore[arg-type]

    confidence_config = layer_configs.pop('confidence', 'linear')
    if confidence_config == 'linear':
      # A single positive scalar controls the sharpness of the Dirichlet
      # posterior around the mean.  Larger values imply lower posterior noise.
      self.layers['confidence'] = tf.keras.Sequential(
          [tf.keras.Input(shape=(encoder_input_dim,)),
           ConstantResponse(
               units=1,
               activation='exp',
           )]
      )
    else:
      self.layers['confidence'] = tf.keras.Sequential.from_config(confidence_config)  # type: ignore[arg-type]

    self.t_vars = tuple(var for layer in self.layers.values() for var in layer.trainable_variables)

    if prior_params is None:
      # A vector prior is the natural unbatched prior for a Dirichlet posterior
      # with event shape ``dim_latent``. 
      concentration = tf.ones((dim_latent,), dtype=tf.float32)
      self.prior = tfp.distributions.Dirichlet(concentration)
    else:
      self.prior = tfp.distributions.Dirichlet(**prior_params)

  def encode_params(self, data: TensorLike) -> tuple[tf.Tensor, tf.Tensor]:
    """Return posterior mean and confidence for concatenated head encodings."""
    data = tf.cast(data, tf.float32)
    mean = tf.clip_by_value(self.layers['mean'](data), self.eps, 1.0)
    mean = mean / tf.reduce_sum(mean, axis=-1, keepdims=True)
    confidence = tf.clip_by_value(self.layers['confidence'](data), 1.0, 100.0)
    return mean, confidence

  def make_encoder(self, data: TensorLike) -> Distribution:
    """Build ``Dirichlet(mean * confidence)`` for ``data``."""
    mean, confidence = self.encode_params(data)
    return tfp.distributions.Dirichlet(mean * confidence)

  def encode(self, data: TensorLike, deterministic: bool = False) -> tf.Tensor:
    """Encode observations into either a Dirichlet sample or posterior mean."""
    if deterministic:
      mean, _ = self.encode_params(data)
      return mean
    return self.make_encoder(data).sample()

  def encode_with_loss(self, data: TensorLike) -> tuple[tf.Tensor, tf.Tensor]:
    """Return a latent sample together with the mean KL-to-prior loss."""
    encoder = self.make_encoder(data)
    sample = encoder.sample()
    loss = tf.reduce_mean(encoder.kl_divergence(self.prior))
    for layer in self.layers.values():
      loss += tf.reduce_sum(layer.losses)
    return sample, loss

  def loss(self, data: TensorLike) -> tf.Tensor:
    """Return the KL divergence loss without sampling a latent value."""
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
            'concentration': self.prior.concentration.numpy().tolist(),
        },
        'layer_configs': {key: layer.get_config() for key, layer in self.layers.items()},
    }

  @staticmethod
  def from_config(config: Mapping[str, Any]) -> 'Categorical':
    """Create a categorical encoder from ``config``."""
    return Categorical(**dict(config))
