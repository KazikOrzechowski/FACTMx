# FACTMx

FACTMx is a TensorFlow / TensorFlow Probability package for building multi-head
variational latent-variable models.  A model combines one shared latent encoder
with one or more modality-specific heads.  Each head knows how to encode its raw
input for the shared encoder and how to decode a shared latent representation
back into a likelihood for its own data type.

## Package layout

```text
FACTMx/
  FACTMx_model.py        Top-level model, ELBO, training loop, save/load helpers
  FACTMx_encoder.py      Linear, attention, and mean-pooling latent encoders
  FACTMx_head.py         Bernoulli, Multinomial, and multivariate Normal heads
  FACTMx_head_GMM.py     Gaussian-mixture observation head
  FACTMx_head_Topic.py   Topic-model observation head
  custom_keras_layers.py Custom serializable Keras layers
  tests.py               Synthetic dataset/configuration helpers
```

## Installation

From the repository root:

```bash
pip install -e .
```

Core dependencies are declared in `setup.py`: `pandas`, `numpy`, `tensorflow`,
`tensorflow-probability`, and `h5py`.

## Conceptual model

A FACTMx model has three stages:

1. **Head encoding**: each head converts one input modality into an
   `encoder_input` tensor.  Some heads also return auxiliary tensors such as
   relaxed assignment samples or assignment logits.
2. **Shared latent encoding**: head encodings are concatenated and passed into a
   shared encoder that returns a distribution over latent coordinates.
3. **Head decoding / loss**: each head defines a likelihood for reconstructing
   its modality from the sampled latent point.  The model combines those
   reconstruction losses with the encoder KL term to form the ELBO.

The top-level objective is implemented in `FACTMx_model.elbo(data)`.  Training
minimizes `-elbo(batch)` in a custom TensorFlow training loop.

## Minimal example

```python
import tensorflow as tf

from FACTMx.FACTMx_model import FACTMx_model
from FACTMx.tests import test_Normal_2D

model_config, dataset, val_data, metadata = test_Normal_2D(sd1=1.0, sd2=0.5)
model_config["optimizer_config"] = tf.keras.optimizers.serialize(
    tf.keras.optimizers.Adam(learning_rate=1e-3)
)

model = FACTMx_model(**model_config)
losses, validation_losses = model.train(dataset, epochs=5, batch_size=200)

# Get deterministic latent coordinates using the posterior mean.
example_batch = next(iter(dataset.batch(32)))
latent_mean = model.get_latent_representation(example_batch)
```

## Configuring heads

Heads are configured with dictionaries.  The `head_type` field selects the
subclass to instantiate through `FACTMx_head.factory(...)`.

```python
heads_config = [
    {"head_type": "Bernoulli", "dim": 1, "head_name": "DNA"},
    {"head_type": "MultiNormal", "dim": 3, "head_name": "RNA"},
]

model = FACTMx_model(dim_latent=2, heads_config=heads_config)
```

Available core heads:

- `Bernoulli`: binary observations.
- `Multinomial`: grouped categorical counts provided as `(observations, counts)`.
- `MultiNormal`: continuous observations with a diagonal multivariate Normal
  decoder.
- `GMM`: sub-batched continuous observations with latent-dependent mixture
  proportions.  Import `FACTMx.FACTMx_head_GMM` before using this type.
- `TopicModel`: document-like count vectors with trainable topic profiles.
  Import `FACTMx.FACTMx_head_Topic` before using this type.

## Configuring encoders

Encoders are selected with `encoder_type` through `FACTMx_encoder.factory(...)`.
If no encoder config is provided, `FACTMx_model` defaults to the linear encoder.

```python
encoder_config = {
    "encoder_type": "Linear",
    "dim_latent": 2,
    "head_dims": [1, 3],
}

model = FACTMx_model(
    dim_latent=2,
    heads_config=heads_config,
    encoder_config=encoder_config,
)
```

Available encoders:

- `Linear`: learned location and diagonal scale networks over concatenated head
  encodings.
- `Attention`: attention aggregation over equally-sized head encodings.  This
  requires all head dimensions to match `dim_latent`.
- `Mean`: mean-pooling over equally-sized head encodings with a learned global
  posterior scale.  This also requires all head dimensions to match
  `dim_latent`.

## Saving and loading

```python
model.save("saved_factmx", overwrite=True, include_optimizer=True)
restored = FACTMx_model.load("saved_factmx", include_optimizer=True)
```

The save directory contains:

- `model_config.json`
- one weight file per encoder/head sub-layer
- optionally `optimizer_state.hdf5`

## Notes for maintainers

- The factories discover subclasses from modules that have already been
  imported.  Import specialised head modules before trying to instantiate
  `GMM` or `TopicModel` from config.
- Several constructors accept either the string `'linear'` for the historic
  default layer or a serialized Keras `Sequential` config.
- The synthetic helpers in `FACTMx/tests.py` return `(model_config, dataset,
  validation_data, metadata)`.  They are examples/smoke-test fixtures rather
  than formal unit tests.
- Some older constructor defaults are mutable dictionaries.  They were left
  unchanged for backward compatibility with the old package, but callers should
  prefer passing fresh config dictionaries.
