"""Topic-model head implementation for FACTMx."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import tensorflow as tf

from FACTMx.head.FACTMx_head import TensorLike
from FACTMx.head.Mixture import Mixture


class Topic(Mixture):
  """Topic-model head for word/count vectors nested within observations."""

  head_type = 'Topic'

  def __init__(
      self,
      dim: int,
      dim_latent: int,
      dim_words: int,
      head_name: str,
      layer_configs: Optional[Mapping[str, Any]] = None,
      topic_profiles: Optional[TensorLike] = None,
      eps: float = 1E-3,
      temperature: float = 1E-4,
  ) -> None:
    super().__init__(
        dim=dim,
        dim_latent=dim_latent,
        head_name=head_name,
        classifier_input_dim=dim_words,
        layer_configs=layer_configs,
        temperature=temperature,
        eps=eps,
        mixture_logits_kernel_initializer='orthogonal',
        mixture_logits_bias_initializer='ones',
        encoder_classifier_bias_initializer='ones',
    )

    self.dim_words = dim_words
    if topic_profiles is None:
      topic_profiles = tf.keras.initializers.RandomNormal()(shape=(dim_words, dim))
    self.topic_profiles_trainable = tf.keras.Variable(topic_profiles, trainable=True, dtype=tf.float32)
    self._set_trainable_variables(self.topic_profiles_trainable)

  def get_log_topic_profiles(self) -> tf.Variable:
    """Return trainable log topic profiles."""
    return self.topic_profiles_trainable

  def get_topic_profiles(self) -> tf.Tensor:
    """Return exponentiated topic profiles."""
    return tf.math.exp(self.get_log_topic_profiles())

  def get_component_log_likelihoods(self, data: TensorLike) -> tf.Tensor:
    """Return topic-level log likelihoods for each nested observation."""
    return tf.matmul(data, self.get_log_topic_profiles())

  def get_config(self) -> dict[str, Any]:
    """Return a JSON-serializable topic-model head configuration."""
    config = super().get_config()
    config.update({
        'head_type': self.head_type,
        'dim_words': self.dim_words,
        'topic_profiles': self.topic_profiles_trainable.numpy().tolist(),
    })
    return config

  @staticmethod
  def from_config(config: Mapping[str, Any]) -> 'Topic':
    """Create a topic-model head from ``config``."""
    config_dict = dict(config)
    config_dict.pop('head_type', None)
    return Topic(**config_dict)
