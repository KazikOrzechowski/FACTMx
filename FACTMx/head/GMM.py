"""Gaussian-mixture-model head implementation for FACTMx."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import tensorflow as tf
import tensorflow_probability as tfp

from FACTMx.head.FACTMx_head import Distribution, TensorLike
from FACTMx.head.Mixture import Mixture


class GMM(Mixture):
  """Gaussian-mixture-model head for sets of continuous sub-observations.

  Args:
    dim: Number of mixture components and encoder proportions.
    dim_latent: Latent representation dimension.
    dim_normal: Feature dimension of each normal sub-observation.
    head_name: Human-readable name for the data modality.
    layer_configs: Optional serialized Keras configs for mixture logits and
      encoder classifier layers.
    mixture_params: Optional initial mixture locations, log-covariances, and
      low-rank covariance perturbation factors.
    temperature: Relaxed-categorical sampling temperature.
    eps: Minimum mixture-proportion floor.
    cov_eps: Diagonal covariance floor.
    max_n_perturb_factor: Maximum low-rank perturbation factor count.
  """

  head_type = 'GMM'

  def __init__(
      self,
      dim: int,
      dim_latent: int,
      dim_normal: int,
      head_name: str,
      layer_configs: Optional[Mapping[str, Any]] = None,
      mixture_params: Optional[Mapping[str, Any]] = None,
      temperature: float = 1E-4,
      eps: float = 1E-3,
      cov_eps: float = 1E-1,
      max_n_perturb_factor: int = 2,
  ) -> None:
    super().__init__(
        dim=dim,
        dim_latent=dim_latent,
        head_name=head_name,
        classifier_input_dim=dim_normal,
        layer_configs=layer_configs,
        temperature=temperature,
        eps=eps,
        mixture_logits_kernel_initializer='random_normal',
        mixture_logits_bias_initializer='ones',
    )

    self.dim_normal = dim_normal
    self.cov_eps = cov_eps
    self.n_cov_perturb_factor = min(dim_normal, max_n_perturb_factor)
    mixture_params = dict(mixture_params or {})

    mixture_locs = mixture_params.pop('loc', 'random')
    if mixture_locs == 'random':
      mixture_locs = tf.keras.initializers.Orthogonal()(shape=(dim, dim_normal))
    self.mixture_locs = tf.keras.Variable(mixture_locs, trainable=True, dtype=tf.float32)

    mixture_diag_covs = mixture_params.pop('cov_diag', 0.1)
    if isinstance(mixture_diag_covs, float):
      mixture_diag_covs = mixture_diag_covs + tf.keras.initializers.Zeros()(shape=(dim, dim_normal))
    self.mixture_diag_covs = tf.keras.Variable(mixture_diag_covs, trainable=True, dtype=tf.float32)

    mixture_cov_perturb = mixture_params.pop('cov_perturb_factor', None)
    if mixture_cov_perturb is None:
      _cov_perturb_shape = (dim, dim_normal, self.n_cov_perturb_factor)
      mixture_cov_perturb = tf.keras.initializers.RandomNormal()(shape=_cov_perturb_shape)
    self.mixture_cov_perturb = tf.keras.Variable(mixture_cov_perturb, trainable=True, dtype=tf.float32)

    self._set_trainable_variables(self.mixture_locs, self.mixture_diag_covs, self.mixture_cov_perturb)

  def get_mixture_distributions(self) -> Distribution:
    """Return the component normal distributions."""
    return tfp.distributions.MultivariateNormalDiagPlusLowRankCovariance(
        self.mixture_locs,
        tf.keras.activations.relu(self.mixture_diag_covs) + self.cov_eps,
        self.mixture_cov_perturb,
    )

  def get_component_log_likelihoods(self, data: TensorLike) -> tf.Tensor:
    """Return Gaussian-mixture log likelihoods for each nested observation."""
    mixtures = self.get_mixture_distributions()
    return mixtures.log_prob(tf.expand_dims(data, -2))

  def get_config(self) -> dict[str, Any]:
    """Return a JSON-serializable GMM head configuration."""
    config = super().get_config()
    config.update({
        'head_type': self.head_type,
        'dim_normal': self.dim_normal,
        'cov_eps': self.cov_eps,
        'max_n_perturb_factor': self.n_cov_perturb_factor,
        'mixture_params': {
            'loc': self.mixture_locs.numpy().tolist(),
            'cov_diag': self.mixture_diag_covs.numpy().tolist(),
            'cov_perturb_factor': self.mixture_cov_perturb.numpy().tolist(),
        },
    })
    return config

  @staticmethod
  def from_config(config: Mapping[str, Any]) -> 'GMM':
    """Create a GMM head from ``config``."""
    config_dict = dict(config)
    if config_dict.get('head_type') == 'GMM':
      config_dict.pop('head_type')
    return GMM(**config_dict)
