"""Base encoder abstractions for FACTMx.

The encoder package turns concatenated head-specific encodings into latent
representations and exposes a small factory API used by :class:`FACTMx_model`.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp

TensorLike = Union[tf.Tensor, np.ndarray]
LayerConfig = Union[str, Mapping[str, Any]]
LayerConfigMap = Optional[Mapping[str, LayerConfig]]
Distribution = tfp.distributions.Distribution


def all_equal(values: Sequence[Any]) -> bool:
  """Return ``True`` when every item in ``values`` is equal."""
  return len(set(values)) == 1


class FACTMx_encoder(tf.Module):
  """Base class for FACTMx encoder modules.

  Args:
    dim_latent: Dimension of the latent representation.
    head_dims: Output dimensions produced by the configured heads.
    name: Optional TensorFlow module name.
  """

  head_dims: Sequence[int]
  dim_latent: int
  layers: Mapping[str, keras.Model]
  t_vars: Sequence[tf.Variable]
  prior: Distribution

  def __init__(self, dim_latent: int, head_dims: Sequence[int], name: Optional[str] = None) -> None:
    super().__init__(name=name)
    self.dim_latent = dim_latent
    self.head_dims = tuple(head_dims)

  def save_weights(self, encoder_path: str) -> None:
    """Save all Keras layer weights using ``encoder_path`` as a prefix."""
    for key, layer in self.layers.items():
      layer.save_weights(f'{encoder_path}_{key}.weights.h5')

  def load_weights(self, encoder_path: str) -> None:
    """Load all Keras layer weights using ``encoder_path`` as a prefix."""
    for key, layer in self.layers.items():
      layer.load_weights(f'{encoder_path}_{key}.weights.h5')

  @staticmethod
  def _recursive_subclasses(cls: type['FACTMx_encoder']) -> list[type['FACTMx_encoder']]:
    """Return all direct and indirect subclasses of ``cls``."""
    subclasses: list[type['FACTMx_encoder']] = []
    for subclass in cls.__subclasses__():
      subclasses.append(subclass)
      subclasses.extend(FACTMx_encoder._recursive_subclasses(subclass))
    return subclasses

  @staticmethod
  def factory(encoder_type: str = 'Linear', **kwargs: Any) -> 'FACTMx_encoder':
    """Instantiate an encoder subclass by its registered ``encoder_type``."""
    encoder_map = {
        encoder.encoder_type: encoder
        for encoder in FACTMx_encoder._recursive_subclasses(FACTMx_encoder)
        if hasattr(encoder, 'encoder_type')
    }
    if encoder_type not in encoder_map:
      available = ', '.join(sorted(encoder_map))
      raise KeyError(f'Unknown encoder_type {encoder_type!r}. Available encoder types: {available}.')
    return encoder_map[encoder_type](**kwargs)
