# FACTMx

FACTMx provides TensorFlow/TensorFlow Probability components for building a
factorized autoencoder with multiple modality-specific heads and a shared latent
encoder.

## Project structure

```text
FACTMx/
├── encoder/
│   ├── __init__.py
│   ├── FACTMx_encoder.py
│   ├── Attention.py
│   ├── Linear.py
│   └── Mean.py
├── head/
│   ├── __init__.py
│   ├── FACTMx_head.py
│   ├── Bernoulli.py
│   ├── ClonalTree.py
│   ├── GMM.py
│   ├── Mixture.py
│   ├── MultiNormal.py
│   ├── Multinomial.py
│   └── Topic.py
├── model.py
├── custom_keras_layers.py
├── math.py
├── utils.py
└── py.typed
```

Each concrete head and encoder lives in its own module and is exported from the
subpackage namespace:

```python
from FACTMx.head import FACTMx_head, Bernoulli, GMM, Mixture, MultiNormal, Multinomial, Topic
from FACTMx.encoder import Attention, Linear, Mean
from FACTMx.model import FACTMx_model
```

Heads can also be created from the default factory used by `FACTMx_model`:

```python
head = FACTMx_head.factory(
    head_type='Topic',
    dim=3,
    dim_latent=2,
    dim_words=100,
    head_name='text',
)
```

`Topic` and `GMM` both inherit from the shared `Mixture` head, which owns the
common mixture-logit layer, encoder-classifier layer, assignment distribution,
mixture-logit decoder, encoder pass, and mixture loss implementation.

## Documentation automation

Install the documentation dependencies and build local HTML docs:

```bash
pip install -e .[docs]
make docs
```

The generated documentation is written to `docs/_build/html`.

A GitHub Actions workflow is included at `.github/workflows/docs.yml` to build
Sphinx documentation on pushes, pull requests, and manual dispatches. The workflow
uploads the generated HTML as a build artifact.

## Type hints

Core model, head, encoder, and custom-layer functions include type annotations
and docstrings. The package includes `FACTMx/py.typed` so downstream static type
checkers can inspect the inline type suggestions.

Run a local type-check pass with:

```bash
pip install -r requirements-dev.txt
make typecheck
```


### Categorical encoder and simple FACTMx heads

This version includes a discrete/simple FACTMx setup for clonal-tree and CRP-like
sequence data.  The intended configuration is:

- `FACTMx.encoder.Categorical`: a Dirichlet variational encoder whose posterior
  mean is a probability vector over latent classes.
- `FACTMx.head.ClonalTreeSimple`: an SNV/binomial head that interprets the
  categorical latent vector as reference-plus-leaf-clone probabilities.
- `FACTMx.head.TopicModelSimple`: a topic/multinomial head that interprets the
  same latent vector as topic probabilities for the one-hot sequence data.

For this setup, the latent dimension should be `2 ** n_levels + 1`, where the
extra class is the reference clone.  Both simple heads expose a `sampling_loss`
attribute: `True` uses relaxed categorical samples in reconstruction losses,
whereas `False` uses the latent probabilities directly.
