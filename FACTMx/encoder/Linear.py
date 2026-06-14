"""Dense encoder implementation for FACTMx."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp

from FACTMx.custom_keras_layers import ConstantResponse
from FACTMx.encoder.FACTMx_encoder import Distribution, FACTMx_encoder, LayerConfigMap, TensorLike


class Linear(FACTMx_encoder):
  """Dense encoder with independent location and diagonal-scale networks."""

  encoder_type = 'Linear'

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
    self.eps = eps
    self.layers: dict[str, keras.Model] = {}
    layer_configs = dict(layer_configs or {})

    loc_config = layer_configs.pop('loc', 'linear')
    if loc_config == 'linear':
      self.layers['loc'] = keras.Sequential(
          [tf.keras.Input(shape=(sum(self.head_dims),)),
           tf.keras.layers.Dense(units=dim_latent, kernel_initializer='orthogonal')]
      )
    else:
      self.layers['loc'] = keras.Sequential.from_config(loc_config)  # type: ignore[arg-type]

    scale_config = layer_configs.pop('scale', 'linear')
    if scale_config == 'linear':
      self.layers['scale'] = tf.keras.Sequential(
          [tf.keras.Input(shape=(sum(self.head_dims),)),
           ConstantResponse(
               units=dim_latent,
               activation='relu',
               bias_initializer={'class_name': 'Constant', 'config': {'value': eps}},
           )]
      )
    else:
      self.layers['scale'] = tf.keras.Sequential.from_config(scale_config)  # type: ignore[arg-type]

    self.t_vars = tuple(var for layer in self.layers.values() for var in layer.trainable_variables)

    if prior_params is None:
      loc = tf.zeros(dim_latent)
      scale_tril = tf.eye(dim_latent)
      self.prior = tfp.distributions.MultivariateNormalTriL(loc, scale_tril)
    else:
      self.prior = tfp.distributions.MultivariateNormalTriL(**prior_params)

  def encode_params(self, data: TensorLike) -> tuple[tf.Tensor, tf.Tensor]:
    """Return latent posterior location and lower-triangular scale tensors."""
    loc = self.layers['loc'](data)
    scale_diag = self.layers['scale'](data) + self.eps
    scale_tril = tf.linalg.diag(scale_diag)
    return loc, scale_tril

  def make_encoder(self, data: TensorLike) -> Distribution:
    """Build the variational posterior distribution for ``data``."""
    loc, scale_tril = self.encode_params(data)
    return tfp.distributions.MultivariateNormalTriL(loc, scale_tril)

  def encode(self, data: TensorLike, deterministic: bool = False) -> tf.Tensor:
    """Encode observations into latent samples or posterior means."""
    if deterministic:
      loc, _ = self.encode_params(data)
      return loc
    return self.make_encoder(data).sample()

  def encode_with_loss(self, data: TensorLike) -> tuple[tf.Tensor, tf.Tensor]:
    """Encode ``data`` and return the sampled latent tensor and KL loss."""
    encoder = self.make_encoder(data)
    sample = encoder.sample()
    loss = tf.reduce_mean(encoder.kl_divergence(self.prior))
    for layer in self.layers.values():
      loss += tf.reduce_sum(layer.losses)
    return sample, loss

  def loss(self, data: TensorLike) -> tf.Tensor:
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
        'layer_configs': {key: layer.get_config() for key, layer in self.layers.items()},
    }

  @staticmethod
  def from_config(config: Mapping[str, Any]) -> 'Linear':
    """Create a linear encoder from ``config``."""
    return Linear(**dict(config))
