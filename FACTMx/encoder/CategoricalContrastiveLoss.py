from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp

from FACTMx.custom_keras_layers import ConstantResponse
from FACTMx.encoder.FACTMx_encoder import Distribution, FACTMx_encoder, LayerConfigMap, TensorLike


class CategoricalContrastiveLoss(FACTMx_encoder):
  encoder_type = 'CategoricalContrastiveLoss'

  def __init__(
      self,
      dim_latent: int,
      head_dims: Sequence[int],
      layer_configs: LayerConfigMap = None,
      name: Optional[str] = None,
      prior_params: Optional[Mapping[str, Any]] = None,
      eps: float = 1E-5,
      straight_through: bool = False,
      pos_pair_scale: float = .01,
      neg_pair_scale: float = .01,
      max_pairs: Optional[int] = None,
      margin: float = .2,
  ) -> None:
    super().__init__(dim_latent, head_dims, name)
    self.eps = float(eps)
    self.straight_through = bool(straight_through)
    self.pos_pair_scale = float(pos_pair_scale)
    self.neg_pair_scale = float(neg_pair_scale)
    self.max_pairs = int(max_pairs)
    self.margin = float(margin)
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

    self.t_vars = tuple(var for layer in self.layers.values() for var in layer.trainable_variables)

    if prior_params is None:
      # A vector prior is the natural unbatched prior for a Dirichlet posterior
      # with event shape ``dim_latent``. 
      concentration = tf.ones((dim_latent,), dtype=tf.float32)
      self.prior = tfp.distributions.Dirichlet(concentration)
    else:
      self.prior = tfp.distributions.Dirichlet(**prior_params)

  def encode_params(self, data: TensorLike) -> tuple[tf.Tensor, tf.Tensor]:
    """Return posterior mean for concatenated head encodings."""
    data = tf.cast(data, tf.float32)
    mean = tf.clip_by_value(self.layers['mean'](data), self.eps, 1.0)
    return mean / tf.reduce_sum(mean, axis=-1, keepdims=True)

  def make_encoder(self, data: TensorLike) -> Distribution:
    """Build ``Dirichlet(mean)`` for ``data``."""
    mean = self.encode_params(data)
    return tfp.distributions.Dirichlet(mean)

  def encode(self, data: TensorLike, deterministic: bool = False) -> tf.Tensor:
    """Encode observations into either a Dirichlet sample or posterior mean."""
    if deterministic:
      mean = self.encode_params(data)
      return mean
    return self.make_encoder(data).sample()

  def trim_pairs(self, pairs: TensorLike, n_init: int) -> tf.Tensor:
    if self.max_pairs is None:
      return pairs, n_init
    inds = []
    for i in tf.unique(pairs[:,0])[0]:
      inds.extend(tf.where(pairs[:,0] == i)[-self.max_pairs:])
    inds = tf.stack(inds)
    return tf.gather(pairs, inds, axis=0), tf.shape(inds)

  def encode_with_loss(self, data: TensorLike, encoder_kwargs) -> tuple[tf.Tensor, tf.Tensor]:
    """Return a latent sample together with the mean KL-to-prior loss."""
    encoder = self.make_encoder(data)
    sample = self.encode_params(data)

    #give straight through assignments or soft assignments
    if self.straight_through:
      soft = sample
      hard = tf.one_hot(tf.argmax(soft, axis=-1), depth=tf.shape(soft)[-1])
      latent_for_heads = tf.stop_gradient(hard - soft) + soft
    else:
      latent_for_heads = sample

    loss = tf.reduce_mean(encoder.kl_divergence(self.prior))
    
    #entropy loss
    usage = tf.reduce_mean(sample, axis=0)
    usage_entropy = -tf.reduce_sum(usage * tf.math.log(usage + self.eps))
    loss -= tf.math.log(usage_entropy + 1E-30) #avoid very low clone usage entropy

    #contrastive loss
    if 'positive_pairs' in encoder_kwargs:
      pair_matrix = tf.linalg.set_diag(encoder_kwargs['positive_pairs'], tf.zeros((n_batch,)))
      positive_ids = tf.where(pair_matrix)
      n_pos = tf.shape(positive_ids)[0]
      if n_pos > 0:
        positive_ids = self.trim_pairs(positive_ids)
        left = tf.gather(sample, positive_ids[:,0], axis=0)
        right = tf.gather(sample, positive_ids[:,1], axis=0)
        symm_kl = tf.reduce_sum(
          (left - right) * (tf.math.log(left) - tf.math.log(right)),
          axis=-1
        )
        loss += self.pos_pair_scale * tf.reduce_mean(symm_kl)
    if 'negative_pairs' in encoder_kwargs:
      pair_matrix = tf.linalg.set_diag(encoder_kwargs['negative_pairs'], tf.zeros((n_batch,)))
      negative_ids = tf.where(pair_matrix)
      n_neg = tf.shape(negative_ids)[0]
      if n_neg > 0:
        negative_ids, n_neg = self.trim_pairs(negative_ids, n_neg)
        left = tf.gather(sample, negative_ids[:,0], axis=0)
        right = tf.gather(sample, negative_ids[:,1], axis=0)
        mid = (left + right) / 2
        js_divergence = left * (tf.math.log(left) - tf.math.log(mid)) + right * (tf.math.log(right) - tf.math.log(mid))
        js_divergence = .5 * tf.reduce_sum(js_divergence, axis=-1)
        push = tf.nn.relu(self.margin - js_divergence)
        loss += self.neg_pair_scale * tf.reduce_mean(push)

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
        'max_pairs': self.max_pairs,
        'pos_pair_scale': self.pos_pair_scale,
        'neg_pair_scale': self.neg_pair_scale,
        'margin': self.margin,
    }

  @staticmethod
  def from_config(config: Mapping[str, Any]) -> 'CategoricalContrastiveLoss':
    """Create a categorical encoder from ``config``."""
    return CategoricalContrastiveLoss(**dict(config))
