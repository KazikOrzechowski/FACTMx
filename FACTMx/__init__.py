"""
FACTMx package metadata.

The package exposes the version and the public symbol list used by setup.py.
Model classes are defined in the sibling modules rather than imported here, so
importing FACTMx stays lightweight.
"""

__version__ = '0.5'
__all__ = ['FACTMx_model', 'FACTMx_encoder', 'FACTMx_head']

