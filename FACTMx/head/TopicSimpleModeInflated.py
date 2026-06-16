"""Mode-inflated simple topic-model head.

``TopicSimpleModeInflated`` extends :class:`FACTMx.head.TopicSimple` with one
additional trainable probability per latent cluster.  With that probability,
observations assigned to a cluster are generated as the cluster's mode sequence
instead of from the ordinary position-wise categorical topic profile.

The class is intentionally small and keeps the public behavior of
``TopicSimple`` wherever possible.  The main additions are:

* ``mode_inflation_logits``: trainable logits of shape ``(dim_latent,)``;
* ``get_mode_inflation_probs()``: returns sigmoid-transformed probabilities;
* a sequence-level reconstruction loss mixing a point mass on each cluster's
  mode sequence with the usual multinomial topic likelihood;
* a decoder that samples either the mode sequence or the ordinary multinomial
  sequence for the sampled/deterministic latent cluster.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import tensorflow as tf
import tensorflow_probability as tfp

from FACTMx.head.FACTMx_head import MultinomialData, TensorLike
from FACTMx.head.TopicSimple import TopicSimple


class TopicSimpleModeInflated(TopicSimple):
  """Simple topic head with cluster-specific mode inflation.

  Args:
    dim_pos: Sequence length/number of positions.
    dim_cat: Number of categories per position.
    dim: Dimension supplied by this head to the shared encoder.
    dim_latent: Number of latent clusters/classes.
    head_name: Name used for variables and serialization.
    mode_inflation_logits: Optional initial logits with shape ``(dim_latent,)``.
      ``sigmoid(mode_inflation_logits[k])`` is the probability that a sequence
      assigned to cluster ``k`` is generated as that cluster's mode sequence.
    **kwargs: Remaining arguments are passed to :class:`TopicSimple`.
  """

  head_type = 'TopicSimpleModeInflated'

  def __init__(
      self,
      dim_pos: int,
      dim_cat: int,
      dim: int,
      dim_latent: int,
      head_name: str,
      mode_inflation_logits: Optional[TensorLike] = None,
      **kwargs: Any,
  ) -> None:
    super().__init__(
        dim_pos=dim_pos,
        dim_cat=dim_cat,
        dim=dim,
        dim_latent=dim_latent,
        head_name=head_name,
        **kwargs,
    )

    if mode_inflation_logits is None:
      mode_inflation_logits = tf.zeros((self.dim_latent,), dtype=tf.float32)
    mode_inflation_logits = tf.cast(mode_inflation_logits, tf.float32)
    if tuple(mode_inflation_logits.shape) != (self.dim_latent,):
      raise ValueError(
          'mode_inflation_logits must have shape '
          f'({self.dim_latent},), got {mode_inflation_logits.shape}.'
      )

    self.mode_inflation_logits = tf.Variable(
        mode_inflation_logits,
        trainable=True,
        dtype=tf.float32,
        name=f'{head_name}_mode_inflation_logits',
    )
    self.t_vars = tuple([*self.t_vars, self.mode_inflation_logits])

  def get_mode_inflation_probs(self) -> tf.Tensor:
    """Return cluster-specific probabilities of observing the mode sequence."""
    return tf.math.sigmoid(self.mode_inflation_logits)

  def _profile_probs(self) -> tf.Tensor:
    """Return normalized profile probabilities used by the decoder/loss.

    ``TopicSimple`` stores profiles directly as a trainable tensor.  This helper
    makes the mode-inflated likelihood robust to small numerical drift by
    clipping and renormalizing before probabilities are used in logs or TFP
    distributions.
    """
    profiles = tf.cast(self.profiles, tf.float32)
    profiles = tf.clip_by_value(profiles, self.eps, 1.0)
    return profiles / tf.reduce_sum(profiles, axis=-1, keepdims=True)

  def _mode_one_hot(self, profiles: tf.Tensor) -> tf.Tensor:
    """Return one-hot mode category per cluster and position."""
    mode_indices = tf.argmax(profiles, axis=-1)
    return tf.one_hot(mode_indices, depth=self.dim_cat, dtype=tf.float32)

  def _sequence_log_prob_by_cluster(
      self,
      observations: TensorLike,
      counts: TensorLike,
  ) -> tf.Tensor:
    """Return mode-inflated sequence log-probability for every cell/cluster.

    Returns a tensor of shape ``(batch, dim_latent)``.  For cluster ``k`` the
    likelihood is

    ``p_mode[k] * 1{x == mode(k)} + (1 - p_mode[k]) * p_topic(x | profile[k])``.
    """
    observations = tf.cast(observations, tf.float32)
    counts = tf.cast(counts, tf.float32)
    profiles = self._profile_probs()

    # Ordinary topic likelihood, summed over sequence positions.
    dist = tfp.distributions.Multinomial(
        total_count=tf.expand_dims(counts, axis=1),
        probs=tf.expand_dims(profiles, axis=0),
    )
    position_log_prob = dist.log_prob(tf.expand_dims(observations, axis=1))
    topic_sequence_log_prob = tf.reduce_sum(position_log_prob, axis=-1)

    # Point mass on the mode sequence for each cluster.
    mode_one_hot = self._mode_one_hot(profiles)
    expected_mode_counts = (
        tf.expand_dims(tf.expand_dims(counts, axis=1), axis=-1)
        * tf.expand_dims(mode_one_hot, axis=0)
    )
    is_mode_position = tf.reduce_all(
        tf.abs(tf.expand_dims(observations, axis=1) - expected_mode_counts) <= self.eps,
        axis=-1,
    )
    is_mode_sequence = tf.reduce_all(is_mode_position, axis=-1)

    mode_probs = tf.clip_by_value(self.get_mode_inflation_probs(), self.eps, 1.0 - self.eps)
    log_mode = tf.math.log(tf.expand_dims(mode_probs, axis=0))
    log_nonmode = (
        tf.math.log1p(-tf.expand_dims(mode_probs, axis=0))
        + topic_sequence_log_prob
    )

    return tf.where(
        is_mode_sequence,
        tf.math.reduce_logsumexp(tf.stack([log_mode, log_nonmode], axis=0), axis=0),
        log_nonmode,
    )

  def loss(self, data: MultinomialData, latent: TensorLike, beta: float = 1) -> tf.Tensor:
    """Return latent-weighted negative mode-inflated sequence log likelihood."""
    del beta
    observations, counts = data
    batch_size = tf.cast(tf.shape(observations)[0], tf.float32)

    log_prob = self._sequence_log_prob_by_cluster(observations, counts)

    if self.sampling_loss:
      assignment = self.get_assignment_distribution(
          tf.math.log(tf.clip_by_value(latent, self.eps, 1.0))
      ).sample()
    else:
      assignment = tf.cast(latent, tf.float32)

    loss = -tf.reduce_sum(log_prob * assignment) / tf.maximum(batch_size, 1.0)

    if self.preencoder:
      loss += tf.reduce_sum(self.layers['preencoder'].losses)

    return loss

  def decode(
      self,
      latent: TensorLike,
      data: MultinomialData,
      deterministic: bool = False,
  ) -> tf.Tensor:
    """Sample observations from the mode-inflated topic decoder.

    A latent cluster is sampled from ``latent`` unless ``deterministic=True``.
    Conditional on that cluster, the decoder samples the cluster's mode sequence
    with probability ``get_mode_inflation_probs()[cluster]`` and otherwise
    samples from the ordinary position-wise multinomial topic profile.
    """
    _observations, counts = data
    latent = tf.cast(latent, tf.float32)
    counts = tf.cast(counts, tf.float32)
    batch_size = tf.shape(latent)[0]

    if deterministic:
      assignment = self.get_deterministic_assignment_sample(latent)
    else:
      logits = tf.math.log(tf.clip_by_value(latent, self.eps, 1.0))
      sampled = tf.squeeze(tf.random.categorical(logits, num_samples=1), axis=-1)
      assignment = tf.one_hot(sampled, depth=self.dim_latent, dtype=tf.float32)

    profiles = self._profile_probs()
    selected_profiles = tf.reduce_sum(
        tf.expand_dims(profiles, axis=0) * assignment[:, :, None, None],
        axis=1,
    )

    multinomial_sample = tfp.distributions.Multinomial(
        total_count=counts,
        probs=selected_profiles,
    ).sample()

    mode_one_hot = self._mode_one_hot(profiles)
    selected_mode = tf.reduce_sum(
        tf.expand_dims(mode_one_hot, axis=0) * assignment[:, :, None, None],
        axis=1,
    )
    mode_sample = counts[:, :, None] * selected_mode

    mode_probs = self.get_mode_inflation_probs()
    selected_mode_probs = tf.reduce_sum(mode_probs[None, :] * assignment, axis=-1)
    if deterministic:
      use_mode = selected_mode_probs >= 0.5
    else:
      use_mode = tf.random.uniform((batch_size,), dtype=tf.float32) < selected_mode_probs

    return tf.where(use_mode[:, None, None], mode_sample, multinomial_sample)

  def get_config(self) -> dict[str, Any]:
    """Return a JSON-serializable mode-inflated topic head configuration."""
    config = super().get_config()
    config['head_type'] = self.head_type
    config['mode_inflation_logits'] = self.mode_inflation_logits.numpy().tolist()
    return config

  @staticmethod
  def from_config(config: Mapping[str, Any]) -> 'TopicSimpleModeInflated':
    """Create a mode-inflated topic head from ``config``."""
    config_dict = dict(config)
    config_dict.pop('head_type', None)
    config_dict.pop('dim_preencoded', None)
    return TopicSimpleModeInflated(**config_dict)


class TopicModelSimpleModeInflated(TopicSimpleModeInflated):
  """Alias head using a TopicModel-style factory name."""

  head_type = 'TopicModelSimpleModeInflated'
