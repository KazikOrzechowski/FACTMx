import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from typing import Tuple, Dict
from FACTMx.FACTMx_head import FACTMx_head
from FACTMx.FACTMx_encoder import FACTMx_encoder

from logging import warning
import json
import os

class FACTMx_model(tf.Module):
  dim_latent: int
  head_dims: Tuple[int]
  heads: Tuple
  encoder: FACTMx_encoder

  def __init__(self, dim_latent,
               heads_config,
               encoder_config=None,
               optimizer_config=None,
               beta=1, prior_params=None,
               name=None):
    super().__init__(name=name)

    self.dim_latent = dim_latent
    self.beta = beta
    self.heads = [FACTMx_head.factory(**head_kwargs, dim_latent=self.dim_latent) for head_kwargs in heads_config]
    self.head_dims = [head.dim for head in self.heads]

    if encoder_config is None:
      self.encoder = FACTMx_encoder(dim_latent, self.head_dims,
                                    prior_params=prior_params)
    else:
      self.encoder = FACTMx_encoder.from_config(encoder_config)

    #gather training variables TODO check why tf.Module fails to collect them automatically
    self.t_vars = (*self.encoder.t_vars, *(var for head in self.heads for var in head.t_vars))

    if optimizer_config is not None:
      self.optimizer = tf.keras.optimizers.get(optimizer_config)
    else:
      self.optimizer = None

  def encode(self, data):
    head_kwargs = [head.encode(data[i]) for i, head in enumerate(self.heads)]
    head_encoded = [head_pass.pop('encoder_input') for head_pass in head_kwargs]
    return self.encoder.encode(tf.concat(head_encoded, axis=1)), head_kwargs

  def decode(self, latent, data):
    return [head.decode(latent, data) for head in self.heads]

  def full_pass(self, data):
    latent, _ = self.encode(data)
    return self.decode(latent, data)

  def elbo(self, data):
    head_kwargs = [head.encode(data[i]) for i, head in enumerate(self.heads)]
    head_encoded = [head_pass.pop('encoder_input') for head_pass in head_kwargs]

    latent, kl_loss = self.encoder.encode_with_loss(tf.concat(head_encoded, axis=-1))

    decoding_losses = [head.loss(data[i],
                                 latent,
                                 beta=self.beta,
                                 **head_kwargs[i])
                          for i, head in enumerate(self.heads)]
    return -tf.math.reduce_mean(tf.stack([kl_loss*self.beta, *decoding_losses], axis=1))

  def train(self,
            dataset,
            validation_dataset=None,
            epochs=1,
            batch_size=200,
            shuffle=True):
    losses = []
    validation_losses = []

    for epoch in range(epochs):
      if shuffle:
        dataset.shuffle(buffer_size=dataset.cardinality())

      batched_dataset = dataset.batch(batch_size)

      for batch in batched_dataset:
        with tf.GradientTape() as tape:
          loss = -self.elbo(batch)
        gradients = tape.gradient(loss, self.t_vars)
        self.optimizer.apply_gradients(zip(gradients, self.t_vars))
        losses.append(loss)

      if validation_dataset is not None:
        validation_losses.append(-self.elbo(validation_dataset))

    return losses, validation_losses

  def get_config(self):
    config = {
        'name': self.name,
        'dim_latent': self.dim_latent,
        'beta': self.beta,
        'heads_config': [head.get_config() for head in self.heads],
        'encoder_config': self.encoder.get_config()
    }
    if self.optimizer is not None:
      config['optimizer_config'] = tf.keras.optimizers.serialize(self.optimizer)

    return config

  def from_config(config):
    for head_config in config['heads_config']:
      head_config.pop('dim_latent')
    return FACTMx_model(**config)

  def save(self, model_path, overwrite=False, include_optimizer=False):
    if os.path.exists(model_path) and not overwrite:
      warning(f'{model_path} exists and overwrite is off. Saving aborted.')
      return

    if not os.path.isdir(model_path):
      os.makedirs(model_path)

    with open(f'{model_path}/model_config.json', 'w') as f:
      config = self.get_config()
      if not include_optimizer:
        config.pop('optimizer_config')
      json.dump(config, f)

    self.encoder.save_weights(f'{model_path}/encoder')
    for i, head in enumerate(model.heads):
      head.save_weights(f'{model_path}/head{i}')

    if include_optimizer:
      optimizer_state = {str(i): v.numpy().tolist() for i, v in enumerate(self.optimizer.variables)}
      with open(f'{model_path}/optimizer_state.json', 'w') as f:
        json.dump(optimizer_state, f)

  def load(model_path, include_optimizer=False):
    with open(f'{model_path}/model_config.json', 'r') as f:
      config = json.load(f)

    if not include_optimizer:
      config.pop('optimizer_config')
    model = FACTMx_model.from_config(config)

    model.encoder.load_weights(f'{model_path}/encoder')
    for i, head in enumerate(model.heads):
      head.load_weights(f'{model_path}/head{i}')

    if include_optimizer:
      with open(f'{model_path}/optimizer_state.json', 'r') as f:
        optimizer_state = json.load(f)
      model.optimizer.build(model.t_vars)
      model.optimizer.load_own_variables(optimizer_state)

    return model