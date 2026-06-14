"""FACTMx package.

The package is organized into ``FACTMx.head`` and ``FACTMx.encoder``
subpackages.
"""

from __future__ import annotations

from typing import Any

__version__ = '0.6.0'

__all__ = [
    'FACTMx_model',
    'FACTMx_encoder',
    'FACTMx_head',
    'Categorical',
    'ClonalTreeSimple',
    'TopicSimple',
    'TopicModelSimple',
]


def __getattr__(name: str) -> Any:
  """Lazily import public classes without importing TensorFlow at setup time."""
  if name == 'FACTMx_model':
    from FACTMx.model import FACTMx_model
    return FACTMx_model
  if name == 'FACTMx_encoder':
    from FACTMx.encoder import FACTMx_encoder
    return FACTMx_encoder
  if name == 'FACTMx_head':
    from FACTMx.head import FACTMx_head
    return FACTMx_head

  if name == 'Categorical':
    from FACTMx.encoder import Categorical
    return Categorical
  if name == 'ClonalTreeSimple':
    from FACTMx.head import ClonalTreeSimple
    return ClonalTreeSimple
  if name == 'TopicSimple':
    from FACTMx.head import TopicSimple
    return TopicSimple
  if name == 'TopicModelSimple':
    from FACTMx.head import TopicModelSimple
    return TopicModelSimple
  raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
