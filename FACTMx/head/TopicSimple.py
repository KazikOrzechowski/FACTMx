"""Simple topic-model head for one-hot/categorical sequence observations.

``TopicSimple`` mirrors ``ClonalTreeSimple``: the shared latent vector is treated
as a soft topic assignment.  The head learns one categorical profile per latent
topic and evaluates sequence positions with a multinomial likelihood.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import tensorflow as tf
import tensorflow_probability as tfp

from FACTMx.head.FACTMx_head import Distribution, FACTMx_head, LayerConfigMap, MultinomialData, TensorLike


class TopicSimple(FACTMx_head):
  """Multinomial topic head driven directly by categorical latent weights.

  Args:
    dim_pos: Sequence length/number of positions.
    dim_cat: Number of categories per position, typically 4 for integer-coded
      nucleotide/state observations.
    dim: Dimension supplied by this head to the shared encoder.  With a
      preencoder this is the preencoder output width; otherwise it must equal
      ``dim_pos * dim_cat``.
    dim_latent: Number of latent topics/classes.  In the simple FACTMx setup it
      is shared with the number of clonal-tree clone classes.
    head_name: Name used for variables and serialization.
    log_profiles: Optional initial topic profiles (pre-softmax) with shape
      ``(dim_latent, dim_pos, dim_cat)``.
    layer_configs: Optional config containing a ``preencoder`` layer.
    eps: Numerical floor used before taking logs.
    temperature: Relaxed categorical sampling temperature.
    encode_logits: If ``True`` and no preencoder is supplied, the encoder input
      is log-normalized empirical category probabilities.  Otherwise raw
      flattened one-hot/count observations are used.
    sampling_loss: If ``True``, reconstruction loss samples a relaxed topic
      assignment from the latent vector.  If ``False``, it uses the latent
      probabilities directly.
  """

  head_type = 'TopicSimple'
  sampling_loss = True

  def __init__(
      self,
      dim_pos: int,
      dim_cat: int,
      dim: int,
      dim_latent: int,
      head_name: str,
      log_profiles: Optional[TensorLike] = None,
      layer_configs: LayerConfigMap = None,
      eps: float = 1E-3,
      temperature: float = 1E-2,
      encode_logits: bool = True,
      sampling_loss: bool = True,
      **kwargs: Any,
  ) -> None:
    del kwargs
    super().__init__(dim, dim_latent, head_name, dim_preencoded=dim_pos * dim_cat)
    self.eps = float(eps)
    self.temperature = float(temperature)
    self.sampling_loss = bool(sampling_loss)
    self.dim_pos = int(dim_pos)
    self.dim_cat = int(dim_cat)
    self.encode_logits = bool(encode_logits)
    layer_configs = dict(layer_configs or {})

    if log_profiles is None:
      log_profiles = tfp.distributions.Dirichlet([.1]*self.dim_cat).sample((self.dim_latent, self.dim_pos))
      log_profiles = tf.math.log(log_profiles + self.eps)
    log_profiles = tf.cast(log_profiles, tf.float32)
    self.log_profiles = tf.Variable(
        log_profiles,
        trainable=True,
        dtype=tf.float32,
        name=f'{head_name}_profiles',
    )

    self.t_vars = tuple([self.log_profiles])

    preencoder_config = layer_configs.pop('preencoder', None)
    if preencoder_config is None:
      self.preencoder = False
      if self.dim != self.dim_pos * self.dim_cat:
        raise ValueError(
            'TopicSimple without a preencoder expects dim == dim_pos * dim_cat.'
        )
    else:
      self.preencoder = True
      if preencoder_config == 'linear':
        self.layers['preencoder'] = tf.keras.Sequential(
            [tf.keras.Input(shape=(self.dim_pos * self.dim_cat,)),
             tf.keras.layers.Dense(units=self.dim, activation='relu')]
        )
      else:
        self.layers['preencoder'] = tf.keras.Sequential.from_config(preencoder_config)  # type: ignore[arg-type]
      self.t_vars += tuple(self.layers['preencoder'].trainable_variables)

      if self.layers['preencoder'].output_shape != (None, self.dim):
        raise ValueError(
            f'TopicSimple preencoder output shape must be (None, {self.dim}), '
            f'got {self.layers["preencoder"].output_shape}.'
        )
      if self.layers['preencoder'].input_shape != (None, self.dim_pos * self.dim_cat):
        raise ValueError(
            'TopicSimple preencoder input shape must be '
            f'(None, {self.dim_pos * self.dim_cat}), got {self.layers["preencoder"].input_shape}.'
        )

  def get_assignment_distribution(self, logits: TensorLike) -> Distribution:
    """Return a relaxed categorical distribution over topics/classes."""
    return tfp.distributions.RelaxedOneHotCategorical(logits=logits, temperature=self.temperature)

  def get_deterministic_assignment_sample(self, logits: TensorLike) -> tf.Tensor:
    """Return hard one-hot topic assignments."""
    return tf.one_hot(tf.argmax(logits, axis=-1), depth=tf.shape(logits)[-1], dtype=tf.float32)

  def get_profiles(self) -> tf.Tensor:
    profiles = tf.math.softmax(self.log_profiles, axis=-1)
    profiles = tf.clip_by_value(profiles, self.eps, 1.0)
    return profiles / tf.reduce_sum(profiles, axis=-1, keepdims=True)
  
  def make_decoder(self, latent: TensorLike, counts: TensorLike, deterministic: bool = False) -> Distribution:
    """Return a per-position multinomial decoder for the topic mixture."""
    latent = tf.cast(latent, tf.float32)
    if deterministic:
      assignment = self.get_deterministic_assignment_sample(latent)
    else:
      assignment = self.get_assignment_distribution(tf.math.log(tf.clip_by_value(latent, self.eps, 1.0))).sample()
    assignment = assignment[:, :, None, None]

    profiles = tf.expand_dims(self.get_profiles(), axis=0)
    probs = tf.reduce_sum(profiles * assignment, axis=1)
    return tfp.distributions.Multinomial(total_count=counts, probs=probs)

  def decode(self, latent: TensorLike, data: MultinomialData) -> tf.Tensor:
    """Sample sequence observations from the topic decoder."""
    _observations, counts = data
    return self.make_decoder(latent, counts).sample()

  def encode(self, data: MultinomialData) -> dict[str, tf.Tensor]:
    """Encode sequence observations as flattened features for the encoder."""
    observations, counts = data
    observations = tf.cast(observations, tf.float32)
    counts = tf.cast(counts, tf.float32)
    if self.preencoder:
      preencoder_input = tf.reshape(observations, shape=(-1, self.dim_pos * self.dim_cat))
      encoder_input = self.layers['preencoder'](preencoder_input)
    elif self.encode_logits:
      denom = tf.maximum(tf.expand_dims(counts, -1), self.eps)
      encoder_input = tf.math.log(observations / denom + self.eps)
      encoder_input = tf.reshape(encoder_input, shape=(-1, self.dim_pos * self.dim_cat))
    else:
      encoder_input = tf.reshape(observations, shape=(-1, self.dim_pos * self.dim_cat))
    return {'encoder_input': encoder_input}

  def loss(self, data: MultinomialData, latent: TensorLike, beta: float = 1) -> tf.Tensor:
    """Return topic-weighted negative multinomial log likelihood."""
    del beta
    observations, counts = data
    counts = tf.expand_dims(tf.cast(counts, tf.float32), axis=1)
    observations = tf.expand_dims(tf.cast(observations, tf.float32), axis=1)
    profiles = tf.expand_dims(self.get_profiles(), axis=0)
    batch_size = tf.cast(tf.shape(observations)[0], tf.float32)

    dist = tfp.distributions.Multinomial(total_count=counts, probs=profiles)
    log_prob = dist.log_prob(observations)

    if self.sampling_loss:
      assignment = self.get_assignment_distribution(tf.math.log(tf.clip_by_value(latent, self.eps, 1.0))).sample()
    else:
      assignment = tf.cast(latent, tf.float32)
    assignment = tf.expand_dims(assignment, axis=-1)
    loss = -tf.reduce_sum(log_prob * assignment) / tf.maximum(batch_size, 1.0)

    if self.preencoder:
      loss += tf.reduce_sum(self.layers['preencoder'].losses)

    return loss

  def get_config(self) -> dict[str, Any]:
    """Return a JSON-serializable TopicSimple head configuration."""
    return {
        'head_type': self.head_type,
        'dim_pos': self.dim_pos,
        'dim_cat': self.dim_cat,
        'dim': self.dim,
        'dim_latent': self.dim_latent,
        'head_name': self.head_name,
        'log_profiles': self.log_profiles.numpy().tolist(),
        'layer_configs': {key: layer.get_config() for key, layer in self.layers.items()},
        'eps': self.eps,
        'temperature': self.temperature,
        'encode_logits': self.encode_logits,
        'sampling_loss': self.sampling_loss,
    }

  @staticmethod
  def from_config(config: Mapping[str, Any]) -> 'TopicSimple':
    """Create a topic-model head from ``config``."""
    config_dict = dict(config)
    config_dict.pop('head_type', None)
    config_dict.pop('dim_preencoded', None)
    return TopicSimple(**config_dict)


class TopicModelSimple(TopicSimple):
  """Alias head using the requested ``TopicModelSimple`` factory name."""

  head_type = 'TopicModelSimple'
