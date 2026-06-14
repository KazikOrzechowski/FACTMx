FACTMx documentation
====================

FACTMx is organized into three main layers:

* ``FACTMx.head`` contains modality-specific heads.
* ``FACTMx.encoder`` contains latent encoders.
* ``FACTMx.model`` composes heads and encoders into a trainable model.

Build the documentation locally with::

   pip install -e .[docs]
   make docs

The generated HTML output is written to ``docs/_build/html``.

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api
