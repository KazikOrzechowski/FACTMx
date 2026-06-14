"""Simple clonal-tree head for mutation/count observations.

``ClonalTreeSimple`` is a lightweight alternative to the full ``ClonalTree``
head.  It does not learn a latent-to-clone classifier.  Instead, the shared
latent vector is expected to already be a probability vector over clone classes,
usually produced by the :class:`FACTMx.encoder.Categorical` encoder.  The head
learns mutation-to-tree-node assignments and evaluates a clone-specific binomial
likelihood for alternate read counts.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import tensorflow as tf
import tensorflow_probability as tfp

from FACTMx.head.FACTMx_head import FACTMx_head, LayerConfigMap, TensorLike

ClonalTreeData = Tuple[TensorLike, TensorLike]
Distribution = tfp.distributions.Distribution


class ClonalTreeSimple(FACTMx_head):
  """Binomial mutation-count head with latent clone probabilities.

  Args:
    dim: Dimension of this head's encoder input.  If a preencoder is supplied,
      this is the preencoder output width.  Without a preencoder it must be
      ``2 * dim_pos`` because the raw encoder input is the concatenation of
      alternate and total read counts.
    dim_latent: Number of latent clone classes.  This must equal
      ``2 ** n_levels + 1``: one reference clone plus all tumour leaf clones.
    dim_pos: Number of mutation/SNV positions.
    n_levels: Depth of the full binary tumour tree.  The number of tumour leaf
      clones is ``2 ** n_levels``.
    head_name: Name used for variables and serialization.
    log_mut_assignment: Optional initial logits assigning each mutation to a
      tree vertex.
    layer_configs: Optional Keras layer config.  If it contains ``preencoder``,
      that layer maps ``concat([mutations, total_counts])`` to ``dim``.
    prob_obs: Alternate-read probability when a mutation is present in a clone.
    prob_unobs: Alternate-read probability when a mutation is absent.
    temperature: Relaxed categorical sampling temperature for clone assignment
      samples used in the reconstruction loss.
    eps: Numerical stability floor for logs/probabilities.
    prop_loss_scale: Multiplicative scale for the reconstruction loss.
    sampling_loss: If ``True``, the loss uses a relaxed sample from the latent
      clone probabilities.  If ``False``, it uses the latent probabilities
      directly.  The Optuna simple-model pipeline tunes this flag.
  """

  head_type = 'ClonalTreeSimple'
  sampling_loss = True

  def __init__(
      self,
      dim: int,
      dim_latent: int,
      dim_pos: int,
      n_levels: int,
      head_name: str,
      log_mut_assignment: Optional[TensorLike] = None,
      layer_configs: LayerConfigMap = None,
      prob_obs: float = 0.5,
      prob_unobs: float = 0.01,
      temperature: float = 1E-2,
      eps: float = 1E-5,
      prop_loss_scale: float = 1.0,
      sampling_loss: bool = True,
  ) -> None:
    self.dim_pos = int(dim_pos)
    super().__init__(dim, dim_latent, head_name, dim_preencoded=2 * self.dim_pos)

    self.n_levels = int(n_levels)
    self.dim_clones = int(2 ** self.n_levels)
    self.dim_clone_classes = self.dim_clones + 1
    self.levels = self.n_levels
    self.prob_obs = float(prob_obs)
    self.prob_unobs = float(prob_unobs)
    self.temperature = float(temperature)
    self.eps = float(eps)
    self.prop_loss_scale = float(prop_loss_scale)
    self.sampling_loss = bool(sampling_loss)
    layer_configs = dict(layer_configs or {})

    if self.dim_latent != self.dim_clone_classes:
      raise ValueError(
          'ClonalTreeSimple expects dim_latent == 2 ** n_levels + 1 so that '
          'the categorical latent vector aligns with reference + leaf clones. '
          f'Got dim_latent={self.dim_latent}, expected {self.dim_clone_classes}.'
      )

    preencoder_config = layer_configs.pop('preencoder', None)
    self.preencoder = preencoder_config is not None
    if not self.preencoder and self.dim != 2 * self.dim_pos:
      raise ValueError(
          'ClonalTreeSimple without a preencoder expects dim == 2 * dim_pos, '
          'because the encoder input is concat([mutations, total_counts]).'
      )

    if self.preencoder:
      if preencoder_config == 'linear':
        self.layers['preencoder'] = tf.keras.Sequential(
            [tf.keras.Input(shape=(2 * self.dim_pos,)),
             tf.keras.layers.Dense(units=self.dim, activation='relu')]
        )
      else:
        self.layers['preencoder'] = tf.keras.Sequential.from_config(preencoder_config)  # type: ignore[arg-type]
      if self.layers['preencoder'].input_shape != (None, 2 * self.dim_pos):
        raise ValueError(
            'ClonalTreeSimple preencoder input shape must be '
            f'(None, {2 * self.dim_pos}), got {self.layers["preencoder"].input_shape}.'
        )
      if self.layers['preencoder'].output_shape != (None, self.dim):
        raise ValueError(
            'ClonalTreeSimple preencoder output shape must be '
            f'(None, {self.dim}), got {self.layers["preencoder"].output_shape}.'
        )

    # The full binary tree has all internal and leaf tumour vertices.  Each
    # mutation chooses one of these vertices; descendants of that vertex inherit
    # the mutation.
    self.all_tumour_clones = int(2 ** (self.n_levels + 1) - 1)
    if log_mut_assignment is None:
      log_mut_assignment = tf.keras.initializers.RandomNormal()(shape=(self.dim_pos, self.all_tumour_clones))

    self.log_mut_assignment = tf.Variable(
        log_mut_assignment,
        trainable=True,
        dtype=tf.float32,
        name=f'{head_name}_log_mut_assignment',
    )
    self.level_shapes = [(self.dim_pos,) + (2,) * i + (1,) * (self.n_levels - i) for i in range(self.n_levels + 1)]
    self.level_inds = [slice(2 ** i - 1, 2 ** (i + 1) - 1) for i in range(self.n_levels + 1)]

    self.t_vars = [self.log_mut_assignment]
    if self.preencoder:
      self.t_vars += list(self.layers['preencoder'].trainable_variables)

  @property
  def current_dim_clone_classes(self) -> int:
    """Return the number of clone classes represented by the latent vector."""
    return self.dim_clone_classes

  def get_assignment_distribution(self, logits: TensorLike) -> Distribution:
    """Return a relaxed categorical distribution over clone assignments."""
    return tfp.distributions.RelaxedOneHotCategorical(logits=logits, temperature=self.temperature)

  def _mutation_assignments_to_clone_profiles(self, mutation_assignments: TensorLike) -> tf.Tensor:
    """Convert mutation-to-tree-node assignments into clone profiles.

    Row 0 of the returned matrix is the reference clone with no mutations.
    Rows 1..K are the tumour leaf clones.
    """
    all_levels = [
        tf.reshape(mutation_assignments[:, inds], shape)
        for shape, inds in zip(self.level_shapes, self.level_inds)
    ]
    all_levels = [tf.broadcast_to(level, self.level_shapes[-1]) for level in all_levels]

    clonal_profiles = tf.reduce_sum(tf.stack(all_levels, axis=0), axis=0)
    clonal_profiles = tf.reshape(clonal_profiles, (self.dim_pos, self.dim_clones))
    reference = tf.zeros((self.dim_pos, 1), dtype=clonal_profiles.dtype)
    clonal_profiles = tf.concat([reference, clonal_profiles], axis=1)
    return tf.transpose(clonal_profiles)

  def get_clone_profiles_sample(self) -> tf.Tensor:
    """Sample relaxed clone profiles from mutation-assignment logits."""
    mut_probs = tf.nn.softmax(self.log_mut_assignment, axis=-1) + self.eps
    mut_assignment_sample = self.get_assignment_distribution(tf.math.log(mut_probs)).sample()
    return self._mutation_assignments_to_clone_profiles(mut_assignment_sample)

  def get_clone_profiles_mle(self) -> tf.Tensor:
    """Return hard MAP clone profiles from mutation-assignment logits."""
    mutation_assignment_mle = tf.one_hot(
        tf.argmax(self.log_mut_assignment, axis=-1),
        depth=self.all_tumour_clones,
        dtype=tf.float32,
    )
    return self._mutation_assignments_to_clone_profiles(mutation_assignment_mle)

  def get_clone_profiles(self, deterministic: bool = False) -> tf.Tensor:
    """Return sampled or deterministic clone profiles."""
    if deterministic:
      return self.get_clone_profiles_mle()
    return self.get_clone_profiles_sample()

  def get_deterministic_assignment_sample(self, logits: TensorLike) -> tf.Tensor:
    """Return hard one-hot assignments from clone probability/logit tensors."""
    return tf.one_hot(tf.argmax(logits, axis=-1), depth=tf.shape(logits)[-1], dtype=tf.float32)

  def get_clone_distributions(self, clonal_profiles: TensorLike, counts: TensorLike) -> Distribution:
    """Return binomial distributions for every cell/clone/mutation triple."""
    probs = clonal_profiles * self.prob_obs + (1 - clonal_profiles) * self.prob_unobs
    probs = tf.expand_dims(probs, axis=0)
    return tfp.distributions.Binomial(total_count=counts, probs=probs)

  def make_decoder(self, latent: TensorLike, data: ClonalTreeData, deterministic: bool = False) -> Distribution:
    """Return the cell-level binomial decoder implied by latent clone weights."""
    _mutations, counts = data
    latent = tf.cast(latent, tf.float32)
    counts = tf.cast(counts, tf.float32)
    counts_by_clone = tf.expand_dims(counts, axis=1)

    if deterministic:
      assignment = self.get_deterministic_assignment_sample(latent)
    else:
      assignment = self.get_assignment_distribution(tf.math.log(tf.clip_by_value(latent, self.eps, 1.0))).sample()
    assignment = tf.expand_dims(assignment, axis=-1)

    clone_profiles = self.get_clone_profiles(deterministic=deterministic)
    probs = self.get_clone_distributions(clone_profiles, counts_by_clone).probs
    probs = tf.reduce_sum(probs * assignment, axis=1)
    return tfp.distributions.Binomial(total_count=counts, probs=probs)

  def decode(self, latent: TensorLike, data: ClonalTreeData, deterministic: bool = False) -> tf.Tensor:
    """Sample alternate-read counts from the decoder."""
    return self.make_decoder(latent, data, deterministic).sample()

  def loss(self, data: ClonalTreeData, latent: TensorLike, beta: float = 1) -> tf.Tensor:
    """Return weighted negative binomial log likelihood for mutation counts."""
    del beta

    mutations, counts = data
    mutations = tf.expand_dims(tf.cast(mutations, tf.float32), axis=1)
    counts = tf.expand_dims(tf.cast(counts, tf.float32), axis=1)
    batch_size = tf.cast(tf.shape(mutations)[0], tf.float32)

    clone_profiles = self.get_clone_profiles(deterministic=False)
    log_like = self.get_clone_distributions(clone_profiles, counts).log_prob(mutations)

    if self.sampling_loss:
      assignment = self.get_assignment_distribution(tf.math.log(tf.clip_by_value(latent, self.eps, 1.0))).sample()
    else:
      assignment = tf.cast(latent, tf.float32)
    assignment = tf.expand_dims(assignment, axis=-1)
    loss = -tf.reduce_sum(log_like * assignment) / tf.maximum(batch_size, 1.0)
    loss *= self.prop_loss_scale

    if self.preencoder:
      loss += tf.reduce_sum(self.layers['preencoder'].losses)

    return loss

  def encode(self, data: ClonalTreeData) -> dict[str, tf.Tensor]:
    """Encode mutation data as ``concat([alternate_counts, total_counts])``."""
    mutations, counts = data
    encoder_input = tf.concat([tf.cast(mutations, tf.float32), tf.cast(counts, tf.float32)], axis=-1)

    if self.preencoder:
      encoder_input = self.layers['preencoder'](encoder_input)

    return {'encoder_input': encoder_input}

  def get_config(self) -> dict[str, Any]:
    """Return a JSON-serializable clonal-tree head configuration."""
    config = super().get_config()
    config.update({
        'head_type': self.head_type,
        'dim_pos': self.dim_pos,
        'n_levels': self.n_levels,
        'prob_obs': self.prob_obs,
        'prob_unobs': self.prob_unobs,
        'temperature': self.temperature,
        'eps': self.eps,
        'prop_loss_scale': self.prop_loss_scale,
        'sampling_loss': self.sampling_loss,
        'layer_configs': {key: layer.get_config() for key, layer in self.layers.items()},
        'log_mut_assignment': self.log_mut_assignment.numpy().tolist(),
    })
    return config

  @staticmethod
  def from_config(config: Mapping[str, Any]) -> 'ClonalTreeSimple':
    """Create a simple clonal-tree head from ``config``."""
    config_dict = dict(config)
    config_dict.pop('head_type', None)
    config_dict.pop('dim_clones', None)
    config_dict.pop('dim_preencoded', None)
    if 'n_levels' not in config_dict and 'levels' in config_dict:
      config_dict['n_levels'] = config_dict.pop('levels')
    else:
      config_dict.pop('levels', None)
    return ClonalTreeSimple(**config_dict)
