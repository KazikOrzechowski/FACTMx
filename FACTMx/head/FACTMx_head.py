"""Base head abstractions for FACTMx.

Heads translate modality-specific observations to encoder inputs and decode
latent representations back to distributions over that modality.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp

TensorLike = Union[tf.Tensor, np.ndarray]
MultinomialData = Tuple[TensorLike, TensorLike]
LayerConfig = Union[str, Mapping[str, Any]]
LayerConfigMap = Optional[Mapping[str, LayerConfig]]
Distribution = tfp.distributions.Distribution


class FACTMx_head(tf.Module):
  """Base class for FACTMx data heads.

  Args:
    dim: Dimension supplied by this head to the shared encoder.
    dim_latent: Latent representation dimension.
    head_name: Human-readable name for the data modality.
    dim_preencoded: Optional raw feature dimension before preencoding.
  """

  dim: int
  dim_preencoded: int
  dim_latent: int
  head_name: str
  layers: dict[str, keras.Model]
  t_vars: Any

  def __init__(self, dim: int, dim_latent: int, head_name: str, dim_preencoded: Optional[int] = None) -> None:
    super().__init__(name=head_name)
    self.dim = dim
    self.dim_latent = dim_latent
    self.dim_preencoded = dim_preencoded if dim_preencoded is not None else dim
    self.head_name = head_name
    self.layers = {}

  def encode(self, data: Any) -> dict[str, Any]:
    """Encode raw head data into keyword arguments consumed by the model."""
    raise NotImplementedError

  def decode(self, latent: TensorLike, data: Any) -> Any:
    """Decode latent tensors into head-specific samples or parameters."""
    raise NotImplementedError

  def save_weights(self, head_path: str) -> None:
    """Save all Keras layer weights using ``head_path`` as a prefix."""
    for key, layer in self.layers.items():
      layer.save_weights(f'{head_path}_{key}.weights.h5')

  def load_weights(self, head_path: str) -> None:
    """Load all Keras layer weights using ``head_path`` as a prefix."""
    for key, layer in self.layers.items():
      layer.load_weights(f'{head_path}_{key}.weights.h5')

  def get_config(self) -> dict[str, Any]:
    """Return the serializable base configuration for the head."""
    return {
        'dim': self.dim,
        'dim_latent': self.dim_latent,
        # 'dim_preencoded': self.dim_preencoded,
        'head_name': self.head_name,
    }

  @staticmethod
  def _recursive_subclasses(cls: type['FACTMx_head']) -> list[type['FACTMx_head']]:
    """Return all direct and indirect subclasses of ``cls``."""
    subclasses: list[type['FACTMx_head']] = []
    for subclass in cls.__subclasses__():
      subclasses.append(subclass)
      subclasses.extend(FACTMx_head._recursive_subclasses(subclass))
    return subclasses

  @staticmethod
  def factory(head_type: str, **kwargs: Any) -> 'FACTMx_head':
    """Instantiate a head subclass by its registered ``head_type``.

    The lookup walks indirect subclasses, so heads derived from intermediate
    abstractions such as :class:`FACTMx.head.Mixture` are available through the
    same factory method as direct :class:`FACTMx_head` subclasses.
    """
    head_map = {
        head.head_type: head
        for head in FACTMx_head._recursive_subclasses(FACTMx_head)
        if hasattr(head, 'head_type')
    }
    if head_type not in head_map:
      available = ', '.join(sorted(head_map))
      raise KeyError(f'Unknown head_type {head_type!r}. Available head types: {available}.')
    return head_map[head_type](**kwargs)
