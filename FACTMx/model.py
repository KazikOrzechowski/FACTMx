"""Top-level FACTMx model composition."""

from __future__ import annotations

import json
import os
from logging import warning
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import tensorflow as tf

from FACTMx.encoder import FACTMx_encoder
from FACTMx.head import FACTMx_head
from FACTMx.utils import select_feature, shuffle_buffer_size

TensorLike = Union[tf.Tensor, np.ndarray]
HeadData = Sequence[Any]


class FACTMx_model(tf.Module):
  """Factorized autoencoder model composed from heads and one encoder.

  Args:
    dim_latent: Dimension of the shared latent representation.
    heads_config: List of head configuration dictionaries accepted by
      ``FACTMx_head.factory``.
    encoder_config: Optional encoder configuration accepted by
      ``FACTMx_encoder.factory``. If omitted, a linear encoder is created.
    optimizer_config: Optional TensorFlow optimizer config/name.
    beta: KL-loss multiplier.
    loss_scales: Optional per-loss scaling vector. The first element scales KL;
      each remaining element scales the matching head loss.
    prior_params: Optional prior parameters for the default encoder.
    name: Optional TensorFlow module name.
  """

  dim_latent: int
  head_dims: Sequence[int]
  heads: Sequence[FACTMx_head]
  encoder: FACTMx_encoder

  def __init__(
      self,
      dim_latent: int,
      heads_config: Sequence[Mapping[str, Any]],
      encoder_config: Optional[Mapping[str, Any]] = None,
      optimizer_config: Optional[Union[str, Mapping[str, Any]]] = None,
      beta: float = 1,
      loss_scales: Optional[Sequence[float]] = None,
      prior_params: Optional[Mapping[str, Any]] = None,
      name: Optional[str] = None,
  ) -> None:
    super().__init__(name=name)

    self.dim_latent = dim_latent
    self.beta = beta
    cleaned_heads_config = [dict(head_config) for head_config in heads_config]
    for head_config in cleaned_heads_config:
      head_config.pop('dim_latent', None)
    self.heads = [FACTMx_head.factory(**head_kwargs, dim_latent=self.dim_latent) for head_kwargs in cleaned_heads_config]
    self.head_dims = [head.dim for head in self.heads]
    self.loss_scales = tf.ones((1 + len(self.heads),)) if loss_scales is None else tf.constant(loss_scales)
    self.layers = None

    if encoder_config is None:
      encoder_config = {
          'encoder_type': 'Linear',
          'dim_latent': dim_latent,
          'head_dims': self.head_dims,
          'prior_params': prior_params,
      }
    self.encoder = FACTMx_encoder.factory(**dict(encoder_config))

    self.t_vars = (*self.encoder.t_vars, *(var for head in self.heads for var in head.t_vars))

    if optimizer_config is not None:
      self.optimizer = tf.keras.optimizers.get(optimizer_config)
    else:
      self.optimizer = None

  def encode(self, data: HeadData) -> Tuple[tf.Tensor, list[dict[str, Any]]]:
    """Encode each head's data and return latent samples plus head kwargs."""
    head_kwargs = [head.encode(data[i]) for i, head in enumerate(self.heads)]
    head_encoded = [head_pass.pop('encoder_input') for head_pass in head_kwargs if 'encoder_input' in head_pass]
    return self.encoder.encode(tf.concat(head_encoded, axis=1)), head_kwargs

  def get_latent_representation(self, data: HeadData) -> tf.Tensor:
    """Return deterministic posterior means for ``data``."""
    head_kwargs = [head.encode(data[i]) for i, head in enumerate(self.heads)]
    head_encoded = [head_pass.pop('encoder_input') for head_pass in head_kwargs if 'encoder_input' in head_pass]

    if isinstance(self.encoder, [CategoricalContrastiveLoss]):
      loc =  self.encoder.encode_params(tf.concat(head_encoded, axis=1))
    else:
      loc, _ = self.encoder.encode_params(tf.concat(head_encoded, axis=1))
    return loc

  def decode(self, latent: TensorLike, data: HeadData) -> list[Any]:
    """Decode latent values for all configured heads."""
    return [head.decode(latent, data[i]) for i, head in enumerate(self.heads)]

  def full_pass(self, data: HeadData) -> list[Any]:
    """Encode and decode data in one forward pass."""
    latent, _ = self.encode(data)
    return self.decode(latent, data)

  def elbo(self, data: HeadData) -> tf.Tensor:
    """Return the evidence lower bound objective for a batch of data."""
    head_kwargs = [head.encode(data[i]) for i, head in enumerate(self.heads)]
    head_encoded = [head_pass.pop('encoder_input') for head_pass in head_kwargs if 'encoder_input' in head_pass]
    encoder_kwargs = {k: v for head_pass in head_kwargs for k, v in head_pass.pop('encoder_kwargs', dict()).items()}

    latent, kl_loss = self.encoder.encode_with_loss(tf.concat(head_encoded, axis=-1), encoder_kwargs)

    decoding_losses = [
        head.loss(data[i], latent, beta=self.beta, **head_kwargs[i])
        for i, head in enumerate(self.heads)
    ]

    all_losses = tf.stack([kl_loss * self.beta, *decoding_losses])
    return -tf.reduce_mean(self.loss_scales * all_losses)

  def update_heads_temperature(self, temperature_update_scale: float) -> None:
    """Multiply each temperature-aware head's temperature by ``scale``."""
    for head in self.heads:
      if 'temperature' in head.__dict__.keys():
        head.temperature *= temperature_update_scale

  def update_heads_eps(self, eps_update_scale: float) -> None:
    """Multiply each epsilon-aware head's epsilon by ``scale``."""
    for head in self.heads:
      if 'eps' in head.__dict__.keys():
        head.eps *= eps_update_scale

  def train(
      self,
      dataset: tf.data.Dataset,
      validation_dataset: Optional[HeadData] = None,
      epochs: int = 1,
      batch_size: int = 200,
      shuffle: bool = True,
      **kwargs: Any,
  ) -> Tuple[list[tf.Tensor], list[tf.Tensor]]:
    """Train the model and return training and validation loss histories."""
    if self.optimizer is None:
      raise ValueError('An optimizer must be configured before calling train().')

    losses: list[tf.Tensor] = []
    validation_losses: list[tf.Tensor] = []

    temperature_update_scale = kwargs.pop('temperature_update', None)
    eps_update_scale = kwargs.pop('eps_update', None)

    for head in self.heads:
      reset_pruning = getattr(head, 'reset_pruning', None)
      if callable(reset_pruning):
        reset_pruning()

    for _epoch in range(epochs):
      if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size(dataset))

      batched_dataset = dataset.batch(batch_size)

      for batch in batched_dataset:
        with tf.GradientTape() as tape:
          loss = -self.elbo(batch)
        gradients = tape.gradient(loss, self.t_vars)
        self.optimizer.apply_gradients(zip(gradients, self.t_vars))
        losses.append(loss)

      if temperature_update_scale is not None:
        self.update_heads_temperature(temperature_update_scale)
      if eps_update_scale is not None:
        self.update_heads_eps(eps_update_scale)

      if validation_dataset is not None:
        validation_datasets = [
            validation_dataset.map(lambda *features, i=i: features[i])
            for i, _ in enumerate(self.heads)
        ]
        batched_validation_datasets = [head_dataset.batch(batch_size) for head_dataset in validation_datasets]
        validation_loss_sum = tf.constant(0.0, dtype=tf.float32)
        validation_batch_count = tf.constant(0.0, dtype=tf.float32)
        for head_batch in zip(*batched_validation_datasets):
          validation_loss_sum += -self.elbo(head_batch)
          validation_batch_count += 1.0
        if validation_batch_count > 0:
          validation_losses.append(validation_loss_sum / validation_batch_count)

    return losses, validation_losses

  def get_config(self) -> dict[str, Any]:
    """Return a JSON-serializable model configuration."""
    config: dict[str, Any] = {
        'name': self.name,
        'dim_latent': self.dim_latent,
        'beta': self.beta,
        'loss_scales': self.loss_scales.numpy().tolist(),
        'heads_config': [head.get_config() for head in self.heads],
        'encoder_config': self.encoder.get_config(),
    }
    if self.optimizer is not None:
      config['optimizer_config'] = tf.keras.optimizers.serialize(self.optimizer)
    return config

  @staticmethod
  def from_config(config: Mapping[str, Any]) -> 'FACTMx_model':
    """Create a FACTMx model from ``config``."""
    config_dict = dict(config)
    config_dict['heads_config'] = [dict(head_config) for head_config in config_dict['heads_config']]
    for head_config in config_dict['heads_config']:
      head_config.pop('dim_latent', None)
    return FACTMx_model(**config_dict)

  def save(self, model_path: str, overwrite: bool = False, include_optimizer: bool = False) -> None:
    """Save model configuration, weights, and optionally optimizer state."""
    if os.path.exists(model_path) and not overwrite:
      warning(f'{model_path} exists and overwrite is off. Saving aborted.')
      return

    if not os.path.isdir(model_path):
      os.makedirs(model_path)

    with open(f'{model_path}/model_config.json', 'w') as f:
      config = self.get_config()
      if not include_optimizer:
        config.pop('optimizer_config', None)
      json.dump(config, f)

    self.encoder.save_weights(f'{model_path}/encoder')
    for i, head in enumerate(self.heads):
      head.save_weights(f'{model_path}/head{i}')

    if include_optimizer:
      with h5py.File(f'{model_path}/optimizer_state.hdf5', 'w') as h5_store:
        for i, v in enumerate(self.optimizer.variables):
          h5_store.create_dataset(name=str(i), data=v.numpy())

  @staticmethod
  def load(model_path: str, include_optimizer: bool = False) -> 'FACTMx_model':
    """Load a model previously saved with :meth:`FACTMx_model.save`."""
    with open(f'{model_path}/model_config.json', 'r') as f:
      config = json.load(f)

    if not include_optimizer:
      config.pop('optimizer_config', None)
    model = FACTMx_model.from_config(config)

    model.encoder.load_weights(f'{model_path}/encoder')
    for i, head in enumerate(model.heads):
      head.load_weights(f'{model_path}/head{i}')

    if include_optimizer:
      with h5py.File(f'{model_path}/optimizer_state.hdf5', 'r') as h5_store:
        model.optimizer.build(model.t_vars)
        model.optimizer.load_own_variables(h5_store)

    return model
