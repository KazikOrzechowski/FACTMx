"""Clonal-tree likelihood head for mutation/count observations."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from FACTMx.head.FACTMx_head import FACTMx_head, LayerConfigMap, TensorLike
from FACTMx.math import categorical_kl_from_logits, np_logsumexp

ClonalTreeData = Tuple[TensorLike, TensorLike]
Distribution = tfp.distributions.Distribution


class ClonalTree(FACTMx_head):
  """Head for modelling mutation observations with a relaxed clonal tree.

  The head assigns mutations to tumour clones arranged across a complete binary
  clonal tree and decodes latent representations into clone-proportion logits.
  Observations are expected as ``(mutations, counts)``, where both tensors have
  shape compatible with ``(batch, dim_pos)``.

  ``dim`` is the number of features passed to the shared encoder. Without a
  ``preencoder`` layer, the encoder input is the clone-assignment logits and
  ``dim`` must equal ``2 ** n_levels + 1`` to include the reference clone. With
  a ``preencoder`` layer, ``dim`` is the preencoder output width and may differ
  from the number of clone classes.

  Args:
    dim: Number of features supplied by this head to the shared encoder.
    dim_latent: Latent representation dimension.
    dim_pos: Number of mutation positions.
    n_levels: Number of binary tumour-clone levels. The number of tumour leaf
      clones is calculated as ``2 ** n_levels``.
    head_name: Human-readable name for this data modality.
    log_mut_assignment: Optional initial log mutation-assignment parameters.
    layer_configs: Optional serialized Keras configs for the ``logits``,
      ``encoder_classifier``, and optional ``preencoder`` layers. Passing
      ``'linear'`` or omitting ``logits``/``encoder_classifier`` uses the
      default linear layers for those components.
    prob_obs: Binomial probability for observed mutations in clone profiles.
    prob_unobs: Binomial probability for unobserved mutations in clone profiles.
    temperature: Relaxed-categorical sampling temperature.
    eps: Numerical floor used for probabilities.
    prop_loss_scale: Multiplicative scale for the clone-proportion KL term.
  """

  head_type = 'ClonalTree'

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
  ) -> None:
    super().__init__(dim, dim_latent, head_name, dim_preencoded=dim_pos)
    self.n_levels = int(n_levels)
    self.dim_clones = int(2 ** self.n_levels)
    self.dim_clone_classes = self.dim_clones + 1
    self.levels = self.n_levels
    self.dim_pos = dim_pos
    self.prob_obs = prob_obs
    self.prob_unobs = prob_unobs
    self.temperature = temperature
    self.eps = eps
    self.prop_loss_scale = prop_loss_scale
    layer_configs = dict(layer_configs or {})

    preencoder_config = layer_configs.pop('preencoder', None)
    self.preencoder = preencoder_config is not None
    if not self.preencoder and self.dim != self.dim_clone_classes:
      raise ValueError(
          'ClonalTree without a preencoder expects dim == 2 ** n_levels + 1 '
          'to include the reference clone. Add a preencoder layer to decouple '
          'the shared encoder dimension from the clone-assignment dimension.'
      )

    logits_config = layer_configs.pop('logits', 'linear')
    if logits_config == 'linear':
      self.layers['logits'] = tf.keras.Sequential(
          [tf.keras.Input(shape=(self.dim_latent,)),
           tf.keras.layers.Dense(units=self.dim_clone_classes, kernel_initializer='orthogonal')]
      )
    else:
      self.layers['logits'] = tf.keras.Sequential.from_config(logits_config)  # type: ignore[arg-type]

    assert self.layers['logits'].output_shape == (None, self.dim_clone_classes)
    assert self.layers['logits'].input_shape == (None, self.dim_latent)

    encoder_classifier_config = layer_configs.pop('encoder_classifier', 'linear')
    if encoder_classifier_config == 'linear':
      self.layers['encoder_classifier'] = tf.keras.Sequential(
          [tf.keras.Input(shape=(2 * self.dim_pos,)),
           tf.keras.layers.Dense(units=self.dim_clone_classes, activation='log_softmax')]
      )
    else:
      self.layers['encoder_classifier'] = tf.keras.Sequential.from_config(encoder_classifier_config)  # type: ignore[arg-type]

    assert self.layers['encoder_classifier'].input_shape == (None, 2 * self.dim_pos)
    assert self.layers['encoder_classifier'].output_shape == (None, self.dim_clone_classes)

    if self.preencoder:
      if preencoder_config == 'linear':
        self.layers['preencoder'] = tf.keras.Sequential(
            [tf.keras.Input(shape=(self.dim_pos,)), tf.keras.layers.Dense(units=self.dim)]
        )
      else:
        self.layers['preencoder'] = tf.keras.Sequential.from_config(preencoder_config)  # type: ignore[arg-type]
      assert self.layers['preencoder'].input_shape == (None, self.dim_pos)
      assert self.layers['preencoder'].output_shape == (None, self.dim)

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

    self.t_vars = [
        *self.layers['logits'].trainable_variables,
        *self.layers['encoder_classifier'].trainable_variables,
        self.log_mut_assignment,
    ]
    if self.preencoder:
      self.t_vars += list(self.layers['preencoder'].trainable_variables)

    self.reset_pruning()

  def reset_pruning(self) -> None:
    """Clear cached tree-pruning state.

    Training should always proceed on the unpruned trainable tree. The pruned
    tree is a post-hoc deterministic view, so this method does not modify any
    trainable variables.
    """
    self.pruned = False
    self.pruned_profiles: Optional[tf.Tensor] = None
    self.pruned_clone_mapping: Optional[tf.Tensor] = None
    self.pruned_intervals: Optional[list[Optional[tuple[int, int]]]] = None
    self.pruned_bic: Optional[float] = None
    self.pruned_log_likelihood: Optional[float] = None
    self.pruned_soft_assignments: Optional[np.ndarray] = None
    self.prune_history: list[dict[str, Any]] = []

  @property
  def current_dim_clone_classes(self) -> int:
    """Return the active number of clone classes, after pruning if set."""
    if self.pruned and self.pruned_profiles is not None:
      return int(self.pruned_profiles.shape[0])
    return self.dim_clone_classes

  def get_assignment_distribution(self, logits: TensorLike) -> Distribution:
    """Return a relaxed categorical distribution over clone assignments."""
    return tfp.distributions.RelaxedOneHotCategorical(logits=logits, temperature=self.temperature)

  def decode_log_probs(self, latent: TensorLike) -> tf.Tensor:
    """Decode latent representations into clone-assignment logits."""
    return self.layers['logits'](latent)

  def _mutation_assignments_to_clone_profiles(self, mutation_assignments: TensorLike) -> tf.Tensor:
    """Convert mutation-to-tree-node assignments into clone profiles.

    Args:
      mutation_assignments: Tensor of shape ``(dim_pos, all_tumour_clones)``
        containing one assignment vector per mutation. The assignment vectors may
        be relaxed probabilities or hard one-hot MLE assignments.

    Returns:
      Tensor of shape ``(dim_clone_classes, dim_pos)``. Row 0 is the fixed
      reference clone with no mutations; remaining rows are tumour leaf clones.
    """
    all_levels = [
        tf.reshape(mutation_assignments[:, inds], shape)
        for shape, inds in zip(self.level_shapes, self.level_inds)
    ]
    all_levels = [tf.broadcast_to(level, self.level_shapes[-1]) for level in all_levels]

    clonal_profiles = tf.reduce_sum(tf.stack(all_levels, axis=0), axis=0)
    clonal_profiles = tf.reshape(clonal_profiles, (self.dim_pos, self.dim_clones))
    clonal_profiles = tf.concat([tf.zeros((self.dim_pos, 1), dtype=clonal_profiles.dtype), clonal_profiles], axis=1)
    return tf.transpose(clonal_profiles)

  def _get_unpruned_clone_profiles_sample(self) -> tf.Tensor:
    """Sample relaxed clone profiles from trainable mutation-assignment logits."""
    mut_probs = tf.nn.softmax(self.log_mut_assignment, axis=-1) + self.eps
    mut_assignment_sample = self.get_assignment_distribution(tf.math.log(mut_probs)).sample()
    return self._mutation_assignments_to_clone_profiles(mut_assignment_sample)

  def _get_unpruned_clone_profiles_mle(self) -> tf.Tensor:
    """Return hard MAP/MLE clone profiles from mutation-assignment logits."""
    mutation_assignment_mle = tf.one_hot(
        tf.argmax(self.log_mut_assignment, axis=-1),
        depth=self.all_tumour_clones,
        dtype=tf.float32,
    )
    return self._mutation_assignments_to_clone_profiles(mutation_assignment_mle)

  def get_pruned_clone_profiles(self) -> tf.Tensor:
    """Return cached pruned clone profiles."""
    if not self.pruned or self.pruned_profiles is None:
      raise ValueError('The ClonalTree head is not pruned. Call prune(...) first.')
    return tf.convert_to_tensor(self.pruned_profiles, dtype=tf.float32)

  def get_clone_profiles_sample(self) -> tf.Tensor:
    """Return sampled profiles, or cached pruned profiles if the head is pruned."""
    if self.pruned:
      return self.get_pruned_clone_profiles()
    return self._get_unpruned_clone_profiles_sample()

  def get_clone_profiles_mle(self) -> tf.Tensor:
    """Return hard MAP/MLE profiles, or cached pruned profiles if pruned."""
    if self.pruned:
      return self.get_pruned_clone_profiles()
    return self._get_unpruned_clone_profiles_mle()

  def get_clone_profiles(self, deterministic: bool = False) -> tf.Tensor:
    """Return active clone profiles.

    If the head has been pruned with :meth:`prune`, the cached pruned profiles
    are returned regardless of ``deterministic``. Otherwise, ``deterministic``
    controls whether the mutation-to-tree-node assignments are hard MAP/MLE or
    relaxed samples.
    """
    if self.pruned:
      return self.get_pruned_clone_profiles()
    if deterministic:
      return self._get_unpruned_clone_profiles_mle()
    return self._get_unpruned_clone_profiles_sample()

  def get_deterministic_assignment_sample(self, logits: TensorLike) -> tf.Tensor:
    """Return a hard one-hot MAP assignment from clone-assignment logits."""
    return tf.one_hot(
        tf.argmax(logits, axis=-1),
        depth=tf.shape(logits)[-1],
        dtype=tf.float32,
    )

  def _build_classifier_input(self, data: ClonalTreeData) -> tf.Tensor:
    """Return concatenated classifier features ``[mutations, total_counts]``."""
    mutations, counts = data
    mutations = tf.convert_to_tensor(mutations, dtype=tf.float32)
    counts = tf.convert_to_tensor(counts, dtype=tf.float32)
    if mutations.shape.rank is not None and mutations.shape.rank != 2:
      raise ValueError('ClonalTree classifier input expects mutations with rank 2.')
    if counts.shape.rank is not None and counts.shape.rank != 2:
      raise ValueError('ClonalTree classifier input expects total counts with rank 2.')
    if mutations.shape[-1] is not None and mutations.shape[-1] != self.dim_pos:
      raise ValueError(f'Expected {self.dim_pos} mutation columns, got {mutations.shape[-1]}.')
    if counts.shape[-1] is not None and counts.shape[-1] != self.dim_pos:
      raise ValueError(f'Expected {self.dim_pos} total-count columns, got {counts.shape[-1]}.')
    tf.debugging.assert_equal(tf.shape(mutations), tf.shape(counts), message='mutations and total_counts must have the same shape')
    return tf.concat([mutations, counts], axis=1)

  def _aggregate_logits_to_pruned_clones(self, logits: TensorLike) -> tf.Tensor:
    """Aggregate original clone logits into pruned clone logits by log-sum-exp."""
    if not self.pruned or self.pruned_clone_mapping is None or self.pruned_profiles is None:
      return tf.convert_to_tensor(logits, dtype=tf.float32)

    logits = tf.convert_to_tensor(logits, dtype=tf.float32)
    mapping = tf.reshape(tf.convert_to_tensor(self.pruned_clone_mapping, dtype=tf.int32), [-1])
    tf.debugging.assert_equal(
        tf.shape(mapping)[0],
        tf.shape(logits)[-1],
        message='pruned_clone_mapping length must match logits last dimension',
    )
    n_pruned = int(self.pruned_profiles.shape[0])
    aggregated = []
    for pruned_idx in range(n_pruned):
      original_indices = tf.where(tf.equal(mapping, pruned_idx))[:, 0]
      selected = tf.gather(logits, original_indices, axis=-1)
      aggregated.append(tf.reduce_logsumexp(selected, axis=-1))
    return tf.stack(aggregated, axis=-1)

  def _aggregate_assignment_weights_to_pruned_clones(self, weights: TensorLike) -> tf.Tensor:
    """Aggregate original clone assignment weights into pruned clone weights."""
    if not self.pruned or self.pruned_clone_mapping is None or self.pruned_profiles is None:
      return tf.convert_to_tensor(weights, dtype=tf.float32)

    weights = tf.convert_to_tensor(weights, dtype=tf.float32)
    mapping = tf.reshape(tf.convert_to_tensor(self.pruned_clone_mapping, dtype=tf.int32), [-1])
    tf.debugging.assert_equal(
        tf.shape(mapping)[0],
        tf.shape(weights)[-1],
        message='pruned_clone_mapping length must match weights last dimension',
    )
    n_pruned = int(self.pruned_profiles.shape[0])
    aggregated = []
    for pruned_idx in range(n_pruned):
      original_indices = tf.where(tf.equal(mapping, pruned_idx))[:, 0]
      selected = tf.gather(weights, original_indices, axis=-1)
      aggregated.append(tf.reduce_sum(selected, axis=-1))
    return tf.stack(aggregated, axis=-1)

  @staticmethod
  def _np_logsumexp(values: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    """Compatibility wrapper around :func:`FACTMx.math.np_logsumexp`."""
    return np_logsumexp(values, axis=axis, keepdims=keepdims)

  @staticmethod
  def _initial_clone_intervals(n_leaves: int) -> list[Optional[tuple[int, int]]]:
    """Return reference + leaf intervals for the original full binary tree."""
    return [None, *[(leaf, leaf + 1) for leaf in range(n_leaves)]]

  @staticmethod
  def _mapping_from_intervals(intervals: list[Optional[tuple[int, int]]], n_leaves: int) -> np.ndarray:
    """Map original clone rows to current pruned clone rows."""
    mapping = np.zeros(n_leaves + 1, dtype=np.int32)
    mapping[0] = 0
    for current_idx, interval in enumerate(intervals):
      if interval is None:
        continue
      start, stop = interval
      mapping[(start + 1):(stop + 1)] = current_idx
    return mapping

  @staticmethod
  def _eligible_prune_pairs(intervals: list[Optional[tuple[int, int]]]) -> list[tuple[int, int]]:
    """Return adjacent sibling-subtree clone rows eligible for one collapse."""
    pairs: list[tuple[int, int]] = []
    for idx in range(1, len(intervals) - 1):
      left = intervals[idx]
      right = intervals[idx + 1]
      if left is None or right is None:
        continue
      left_start, left_stop = left
      right_start, right_stop = right
      left_size = left_stop - left_start
      right_size = right_stop - right_start
      if left_size != right_size:
        continue
      if left_stop != right_start:
        continue
      parent_size = 2 * left_size
      if left_start % parent_size != 0:
        continue
      pairs.append((idx, idx + 1))
    return pairs

  @staticmethod
  def _collapse_prune_pair_np(
      profiles: np.ndarray,
      intervals: list[Optional[tuple[int, int]]],
      pair: tuple[int, int],
  ) -> tuple[np.ndarray, list[Optional[tuple[int, int]]]]:
    """Collapse two sibling clone columns into their parent union profile."""
    left_idx, right_idx = pair
    left_interval = intervals[left_idx]
    right_interval = intervals[right_idx]
    if left_interval is None or right_interval is None:
      raise ValueError('The reference clone cannot be pruned.')

    collapsed_profile = np.maximum(profiles[left_idx], profiles[right_idx])
    collapsed_interval = (left_interval[0], right_interval[1])

    new_profiles = []
    new_intervals: list[Optional[tuple[int, int]]] = []
    for idx in range(profiles.shape[0]):
      if idx == left_idx:
        new_profiles.append(collapsed_profile)
        new_intervals.append(collapsed_interval)
      elif idx == right_idx:
        continue
      else:
        new_profiles.append(profiles[idx])
        new_intervals.append(intervals[idx])

    return np.stack(new_profiles, axis=0), new_intervals

  def _clone_log_likelihood_np(
      self,
      mutations: np.ndarray,
      counts: np.ndarray,
      profiles: np.ndarray,
  ) -> np.ndarray:
    """Return cell x clone binomial log likelihoods, excluding constants."""
    profiles = np.asarray(profiles, dtype=float)
    mutations = np.asarray(mutations, dtype=float)
    counts = np.asarray(counts, dtype=float)
    reference_counts = counts - mutations
    if np.any(reference_counts < -1e-6):
      raise ValueError('Mutation/alternate counts cannot exceed total counts.')
    reference_counts = np.maximum(reference_counts, 0.0)

    p_obs = float(np.clip(self.prob_obs, self.eps, 1.0 - self.eps))
    p_unobs = float(np.clip(self.prob_unobs, self.eps, 1.0 - self.eps))
    probs = profiles[None, :, :] * p_obs + (1.0 - profiles[None, :, :]) * p_unobs
    probs = np.clip(probs, self.eps, 1.0 - self.eps)

    return np.sum(
        mutations[:, None, :] * np.log(probs)
        + reference_counts[:, None, :] * np.log1p(-probs),
        axis=-1,
    )

  @staticmethod
  def _aggregate_log_prior_np(
      original_log_prior: np.ndarray,
      mapping: np.ndarray,
      n_pruned: int,
  ) -> np.ndarray:
    """Aggregate original clone log priors into pruned clone log priors."""
    aggregated = np.full((original_log_prior.shape[0], n_pruned), -np.inf, dtype=float)
    for original_idx, pruned_idx in enumerate(mapping):
      aggregated[:, pruned_idx] = np.logaddexp(aggregated[:, pruned_idx], original_log_prior[:, original_idx])
    normalizer = ClonalTree._np_logsumexp(aggregated, axis=1, keepdims=True)
    return aggregated - normalizer

  def _score_pruned_profiles_np(
      self,
      mutations: np.ndarray,
      counts: np.ndarray,
      profiles: np.ndarray,
      intervals: list[Optional[tuple[int, int]]],
      original_log_prior: np.ndarray,
  ) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Return BIC-style score and posterior weights for a candidate pruning."""
    mapping = self._mapping_from_intervals(intervals, self.dim_clones)
    log_prior = self._aggregate_log_prior_np(original_log_prior, mapping, profiles.shape[0])
    log_like = self._clone_log_likelihood_np(mutations, counts, profiles)
    log_post = log_prior + log_like
    log_post = log_post - self._np_logsumexp(log_post, axis=1, keepdims=True)
    posterior = np.exp(log_post)

    weighted_log_likelihood = float(np.sum(posterior * log_like))
    n_observations = max(1, int(np.prod(mutations.shape)))
    n_non_reference_clones = profiles.shape[0] - 1
    n_parameters = int(n_non_reference_clones * self.dim_pos)
    bic_score = float(2.0 * weighted_log_likelihood - n_parameters * np.log(n_observations))
    return bic_score, weighted_log_likelihood, posterior, log_like, mapping

  def prune(
      self,
      data: ClonalTreeData,
      min_clone: int = 1,
      bic_tolerance: float = 0.0,
  ) -> dict[str, Any]:
    """Prune the deterministic clonal tree by greedily optimizing BIC.

    Args:
      data: Tuple ``(alternate_counts, total_counts)`` with shape
        ``(n_cells, dim_pos)`` for each tensor.
      min_clone: Minimum number of non-reference clones retained after pruning.
        The reference clone is fixed at row 0 and is never pruned.
      bic_tolerance: Required improvement in the higher-is-better BIC score.

    Returns:
      Dictionary with the pruned profiles, original-to-pruned clone mapping,
      BIC trace, soft posterior assignments, and collapse history.

    Notes:
      This method is post-hoc: it does not change trainable variables such as
      ``log_mut_assignment`` or decoder/classifier layer weights. It stores a
      pruned view that is used by ``decode`` and the profile accessors until
      ``reset_pruning`` is called. ``FACTMx_model.train`` calls
      ``reset_pruning`` automatically before fitting.
    """
    if min_clone < 1:
      raise ValueError('min_clone must be at least 1.')
    if min_clone > self.dim_clones:
      raise ValueError('min_clone cannot exceed the number of tumour leaf clones.')

    mutations_tf = tf.convert_to_tensor(data[0], dtype=tf.float32)
    counts_tf = tf.convert_to_tensor(data[1], dtype=tf.float32)
    mutations = mutations_tf.numpy()
    counts = counts_tf.numpy()

    if mutations.ndim != 2 or counts.ndim != 2:
      raise ValueError('ClonalTree.prune expects two matrices of shape n_cells x dim_pos.')
    if mutations.shape != counts.shape:
      raise ValueError('alternate_counts and total_counts must have the same shape.')
    if mutations.shape[1] != self.dim_pos:
      raise ValueError(f'Expected {self.dim_pos} mutation columns, got {mutations.shape[1]}.')
    if np.any(mutations < 0) or np.any(counts < 0):
      raise ValueError('Counts must be non-negative.')
    if np.any(mutations > counts):
      raise ValueError('Alternate counts cannot exceed total counts.')

    # Pruning always starts from the deterministic unpruned tree.
    original_profiles = self._get_unpruned_clone_profiles_mle().numpy().astype(float)
    original_profiles = (original_profiles > 0.5).astype(np.float32)
    intervals = self._initial_clone_intervals(self.dim_clones)
    profiles = original_profiles.copy()

    original_log_prior = self.layers['encoder_classifier'](self._build_classifier_input((mutations_tf, counts_tf))).numpy()
    original_log_prior = original_log_prior - self._np_logsumexp(original_log_prior, axis=1, keepdims=True)

    current_bic, current_ll, posterior, log_like, mapping = self._score_pruned_profiles_np(
        mutations, counts, profiles, intervals, original_log_prior
    )
    bic_trace = [current_bic]
    history: list[dict[str, Any]] = []

    while profiles.shape[0] - 1 > min_clone:
      candidates = self._eligible_prune_pairs(intervals)
      if not candidates:
        break

      best_candidate: Optional[dict[str, Any]] = None
      for pair in candidates:
        candidate_profiles, candidate_intervals = self._collapse_prune_pair_np(profiles, intervals, pair)
        candidate_bic, candidate_ll, candidate_posterior, candidate_log_like, candidate_mapping = self._score_pruned_profiles_np(
            mutations, counts, candidate_profiles, candidate_intervals, original_log_prior
        )
        if best_candidate is None or candidate_bic > best_candidate['bic']:
          best_candidate = {
              'pair': pair,
              'profiles': candidate_profiles,
              'intervals': candidate_intervals,
              'bic': candidate_bic,
              'log_likelihood': candidate_ll,
              'posterior': candidate_posterior,
              'log_like': candidate_log_like,
              'mapping': candidate_mapping,
          }

      if best_candidate is None:
        break
      if best_candidate['bic'] <= current_bic + bic_tolerance:
        break

      left_idx, right_idx = best_candidate['pair']
      left_interval = intervals[left_idx]
      right_interval = intervals[right_idx]
      profiles = best_candidate['profiles']
      intervals = best_candidate['intervals']
      current_bic = float(best_candidate['bic'])
      current_ll = float(best_candidate['log_likelihood'])
      posterior = best_candidate['posterior']
      log_like = best_candidate['log_like']
      mapping = best_candidate['mapping']
      bic_trace.append(current_bic)

      history.append({
          'collapsed_clone_rows': (int(left_idx), int(right_idx)),
          'collapsed_original_leaf_interval': (
              int(left_interval[0]), int(right_interval[1])  # type: ignore[index]
          ),
          'bic': current_bic,
          'log_likelihood': current_ll,
          'n_non_reference_clones': int(profiles.shape[0] - 1),
      })

    self.pruned = True
    self.pruned_profiles = tf.convert_to_tensor(profiles, dtype=tf.float32)
    self.pruned_clone_mapping = tf.convert_to_tensor(mapping, dtype=tf.int32)
    self.pruned_intervals = intervals
    self.pruned_bic = float(current_bic)
    self.pruned_log_likelihood = float(current_ll)
    self.pruned_soft_assignments = posterior.copy()
    self.prune_history = history

    return {
        'pruned': True,
        'pruned_profiles': profiles.astype(np.int8),
        'original_to_pruned_clone_mapping': mapping.copy(),
        'pruned_intervals': intervals,
        'bic': float(current_bic),
        'bic_higher_is_better': float(current_bic),
        'log_likelihood': float(current_ll),
        'bic_trace': np.asarray(bic_trace, dtype=float),
        'soft_assignments': posterior.copy(),
        'log_likelihood_matrix': log_like.copy(),
        'prune_history': history,
        'n_non_reference_clones': int(profiles.shape[0] - 1),
    }

  def get_clone_distributions(self, clonal_profiles: TensorLike, counts: TensorLike) -> Distribution:
    """Return binomial distributions for mutation counts in each clone."""
    probs = clonal_profiles * self.prob_obs + (1 - clonal_profiles) * self.prob_unobs
    probs = tf.expand_dims(probs, axis=0)
    return tfp.distributions.Binomial(total_count=counts, probs=probs)

  def decode(
      self,
      latent: TensorLike,
      data: ClonalTreeData,
      sample: bool = True,
      deterministic: bool = False,
  ) -> tuple[Optional[tf.Tensor], tf.Tensor, tf.Tensor]:
    """Decode latent values into clone-assignment samples and likelihood terms.

    Args:
      latent: Latent samples or deterministic latent representations.
      data: Tuple of mutation observations and total counts.
      sample: If ``True``, return a clone-assignment tensor. If ``False``,
        return ``None`` in the sample slot while still returning likelihood and
        prior-logit terms.
      deterministic: If ``True``, use hard MAP/MLE mutation-to-tree-node
        assignments to construct clone profiles and return hard one-hot MAP
        observation assignments. If ``False``, keep the previous stochastic
        relaxed-categorical behavior.
    """
    mutations, counts = data
    mutations = tf.expand_dims(mutations, axis=1)
    counts = tf.expand_dims(counts, axis=1)

    log_prior = self._aggregate_logits_to_pruned_clones(self.decode_log_probs(latent))
    clone_profiles = self.get_clone_profiles(deterministic=deterministic)
    log_like = self.get_clone_distributions(clone_profiles, counts).log_prob(mutations)
    log_like = tf.reduce_sum(log_like, axis=-1)
    log_post = log_prior + log_like

    if not sample:
      assignment_sample = None
    elif deterministic:
      assignment_sample = self.get_deterministic_assignment_sample(log_post)
    else:
      assignment_sample = self.get_assignment_distribution(log_post).sample()

    return assignment_sample, log_like, log_prior

  def loss(
      self,
      data: ClonalTreeData,
      latent: TensorLike,
      encoder_assignment_logits: TensorLike,
      encoder_assignment_sample: TensorLike,
      beta: float = 1,
  ) -> tf.Tensor:
    """Return the clonal-tree reconstruction and assignment KL loss."""
    del beta
    batch_size = tf.cast(tf.shape(data[0])[0], tf.float32)
    _assignment_sample, log_like, log_probs = self.decode(latent, data, sample=False)
    encoder_assignment_logits = self._aggregate_logits_to_pruned_clones(encoder_assignment_logits)
    encoder_assignment_sample = self._aggregate_assignment_weights_to_pruned_clones(encoder_assignment_sample)

    kl_loss = tf.reduce_sum(categorical_kl_from_logits(encoder_assignment_logits, log_probs))
    log_loss = tf.reduce_sum(encoder_assignment_sample * log_like)

    loss_terms = [
        kl_loss / batch_size * self.prop_loss_scale,
        log_loss / batch_size,
        *self.layers['logits'].losses,
        *self.layers['encoder_classifier'].losses,
    ]
    if self.preencoder:
      loss_terms.extend(self.layers['preencoder'].losses)

    return tf.reduce_sum(loss_terms)

  def encode(self, data: ClonalTreeData) -> dict[str, tf.Tensor]:
    """Encode mutation observations into encoder inputs and clone assignments."""
    mutations, _counts = data
    mutations = tf.convert_to_tensor(mutations, dtype=tf.float32)
    classifier_input = self._build_classifier_input(data)
    assignment_logits = self.layers['encoder_classifier'](classifier_input)
    assignment_logits = tf.math.log(tf.nn.softmax(assignment_logits, axis=-1) + self.eps)
    assignment_sample = self.get_assignment_distribution(assignment_logits).sample()

    if self.preencoder:
      encoder_input = self.layers['preencoder'](mutations)
    else:
      encoder_input = assignment_logits

    return {
        'encoder_input': encoder_input,
        'encoder_assignment_logits': assignment_logits,
        'encoder_assignment_sample': assignment_sample,
    }

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
        'layer_configs': {key: layer.get_config() for key, layer in self.layers.items()},
        'log_mut_assignment': self.log_mut_assignment.numpy().tolist(),
    })
    return config

  @staticmethod
  def from_config(config: Mapping[str, Any]) -> 'ClonalTree':
    """Create a clonal-tree head from ``config``."""
    config_dict = dict(config)
    config_dict.pop('head_type', None)
    config_dict.pop('dim_clones', None)
    config_dict.pop('dim_preencoded', None)
    if 'n_levels' not in config_dict and 'levels' in config_dict:
      config_dict['n_levels'] = config_dict.pop('levels')
    else:
      config_dict.pop('levels', None)
    return ClonalTree(**config_dict)
