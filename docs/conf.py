"""Sphinx configuration for FACTMx documentation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

project = 'FACTMx'
author = 'Kazimierz Oksza-Orzechowski'
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
]

autosummary_generate = True
autodoc_typehints = 'description'
autodoc_member_order = 'bysource'
napoleon_google_docstring = True
napoleon_numpy_docstring = True
html_theme = 'furo'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
