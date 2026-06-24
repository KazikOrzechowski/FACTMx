from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp

from FACTMx.custom_keras_layers import ConstantResponse
from FACTMx.encoder.FACTMx_encoder import Distribution, FACTMx_encoder, LayerConfigMap, TensorLike


class CategoricalContrastiveLoss(FACTMx_encoder):
  encoder_type = 'CategoricalContrastiveLoss'
  max_confidence = 1.

  def __init__(
      self,
      dim_latent: int,
      head_dims: Sequence[int],
      layer_configs: LayerConfigMap = None,
      name: Optional[str] = None,
      prior_params: Optional[Mapping[str, Any]] = None,
      eps: float = 1E-5,
      pos_pair_scale: float = .01,
      neg_pair_scale: float = .01,
      margin: float = .2,
  ) -> None:
    super().__init__(dim_latent, head_dims, name)
    self.eps = float(eps)
    self.pos_pair_scale = pos_pair_scale
    self.neg_pair_scale = neg_pair_scale
    self.margin = margin
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
    confidence = tf.clip_by_value(self.layers['confidence'](data), 1.0, 1.0)

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

  def encode_with_loss(self, data: TensorLike, encoder_kwargs) -> tuple[tf.Tensor, tf.Tensor]:
    """Return a latent sample together with the mean KL-to-prior loss."""
    n_batch, _ = data.shape
    encoder = self.make_encoder(data)
    sample, _ = self.encode_params(data)

    #rewrite
    soft = sample
    hard = tf.one_hot(tf.argmax(soft, axis=-1), depth=tf.shape(soft)[-1])
    latent_for_heads = tf.stop_gradient(hard - soft) + soft

    loss = tf.reduce_mean(encoder.kl_divergence(self.prior))
    
    #entropy loss
    usage = tf.reduce_mean(sample, axis=0)
    usage_entropy = -tf.reduce_sum(usage * tf.math.log(usage + self.eps))
    loss -= tf.math.log(usage_entropy + 1E-300) #avoid very low clone usage entropy

    #contrastive loss
    if 'positive_pairs' in encoder_kwargs:
      pair_matrix = tf.linalg.set_diag(encoder_kwargs['positive_pairs'], tf.zeros((n_batch,)))
      positive_ids = tf.where(pair_matrix)
      n_pos = positive_ids.shape[0]
      if n_pos > 0:
        left = tf.gather(sample, positive_ids[:,0], axis=0)
        right = tf.gather(sample, positive_ids[:,1], axis=0)
        symm_kl = tf.reduce_sum((left - right) * (tf.math.log(left) - tf.math.log(right)))
        loss += self.pos_pair_scale * symm_kl / n_pos
    if 'negative_pairs' in encoder_kwargs:
      pair_matrix = tf.linalg.set_diag(encoder_kwargs['negative_pairs'], tf.zeros((n_batch,)))
      negative_ids = tf.where(pair_matrix)
      n_neg = negative_ids.shape[0]
      if n_neg > 0:
        left = tf.gather(sample, negative_ids[:,0], axis=0)
        right = tf.gather(sample, negative_ids[:,1], axis=0)
        mid = (left + right) / 2
        js_divergence = left * (tf.math.log(left) - tf.math.log(mid)) + right * (tf.math.log(right) - tf.math.log(mid))
        js_divergence = .5 * tf.reduce_sum(js_divergence, axis=-1)
        push = tf.nn.relu(self.margin - js_divergence)
        loss += self.neg_pair_scale * tf.reduce_sum(push) / n_neg

    #regularization losses
    for layer in self.layers.values():
      loss += tf.reduce_sum(layer.losses)
    return latent_for_heads, loss

  def loss(self, data: TensorLike, encoder_kwargs=None) -> tf.Tensor:
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
        'pos_pair_scale': self.pos_pair_scale,
        'neg_pair_scale': self.neg_pair_scale,
        'margin': self.margin,
    }

  @staticmethod
  def from_config(config: Mapping[str, Any]) -> 'CategoricalContrastiveLoss':
    """Create a categorical encoder from ``config``."""
    return CategoricalContrastiveLoss(**dict(config))
