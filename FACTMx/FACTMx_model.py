import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from typing import Tuple, Dict
from FACTMx.FACTMx_head import FACTMx_head
from FACTMx.FACTMx_encoder import FACTMx_encoder

import multiprocessing
from logging import warning
import json
import h5py
import os



class FACTMx_model(tf.Module):
  dim_latent: int
  head_dims: Tuple[int]
  heads: Tuple
  encoder: FACTMx_encoder

  def __init__(self, dim_latent, dim_first_pass,
               heads_config,
               encoder_config=None,
               optimizer_config=None,
               beta=1, loss_scales=None,
               prior_params=None,
               name=None):
    super().__init__(name=name)

    self.dim_latent = dim_latent
    self.dim_first_pass = dim_first_pass
    self.beta = beta
    for head_config in heads_config:
      head_config.pop('dim_latent', None)
      head_config.pop('dim_first_pass', None)
    self.heads = [FACTMx_head.factory(**head_kwargs, dim_latent=self.dim_latent, dim_first_self.dim_first_pass) for head_kwargs in heads_config]
    self.head_dims = [head.dim for head in self.heads]
    self.loss_scales = tf.ones((1+len(self.heads),)) if loss_scales is None else tf.constant(loss_scales)
    self.layers = None #handled by pruning module

    if encoder_config is None:
      encoder_config = {'encoder_type': 'Linear',
                        'dim_latent': dim_latent, 
                        'head_dims': self.head_dims,
                        'prior_params': prior_params}
    self.encoder = FACTMx_encoder.factory(**encoder_config)
    self.first_encoder = FACTMx_encoder.factory(encoder_type='Linear',
                                                dim_latent=dim_latent, 
                                                head_dims=self.head_dims,
                                                prior_params=prior_params)

    #gather training variables TODO check why tf.Module fails to collect them automatically
    self.t_vars = (*self.encoder.t_vars, *self.first_encoder.t_vars, *(var for head in self.heads for var in head.t_vars))

    if optimizer_config is not None:
      self.optimizer = tf.keras.optimizers.get(optimizer_config)
    else:
      self.optimizer = None

  def encode(self, data):
    head_first_pass = [head.first_pass(data[i]) for i, head in enumerate(self.heads)]
    head_first_encoded = [head_pass.pop('encoder_input') for head_pass in head_first_pass]
    encoder_first_pass = self.first_encoder.encode(tf.concat(head_first_encoded, axis=1))
    
    head_kwargs = [head.encode(data[i], encoder_first_pass) for i, head in enumerate(self.heads)]
    head_encoded = [head_pass.pop('encoder_input') for head_pass in head_kwargs]
    return self.encoder.encode(tf.concat(head_encoded, axis=1)), head_kwargs

  def get_latent_representation(self, data):
    #deterministic encoder
    head_first_pass = [head.first_pass(data[i]) for i, head in enumerate(self.heads)]
    head_first_encoded = [head_pass.pop('encoder_input') for head_pass in head_first_pass]
    encoder_first_pass = self.first_encoder.encode_params(tf.concat(head_first_encoded, axis=1))
    
    head_kwargs = [head.encode(data[i], encoder_first_pass) for i, head in enumerate(self.heads)]
    head_encoded = [head_pass.pop('encoder_input') for head_pass in head_kwargs]

    loc, _ = self.encoder.encode_params(tf.concat(head_encoded, axis=1))
    return loc

  def decode(self, latent, data):
    return [head.decode(latent, data[i]) for i, head in enumerate(self.heads)]

  def full_pass(self, data):
    latent, _ = self.encode(data)
    return self.decode(latent, data)

  def elbo(self, data):
    head_first_pass = [head.first_pass(data[i]) for i, head in enumerate(self.heads)]
    head_first_encoded = [head_pass.pop('encoder_input') for head_pass in head_first_pass]
    encoder_first_pass = self.first_encoder.encode(tf.concat(head_first_encoded, axis=1))
    
    head_kwargs = [head.encode(data[i], encoder_first_pass) for i, head in enumerate(self.heads)]
    head_encoded = [head_pass.pop('encoder_input') for head_pass in head_kwargs]

    latent, kl_loss = self.encoder.encode_with_loss(tf.concat(head_encoded, axis=-1))

    decoding_losses = [head.loss(data[i],
                                 latent,
                                 beta=self.beta,
                                 **head_kwargs[i])
                          for i, head in enumerate(self.heads)]
    
    all_losses = tf.stack([kl_loss*self.beta, *decoding_losses])
    return -tf.reduce_mean(self.loss_scales * all_losses)

  def update_heads_temperature(self, temperature_update_scale):
    for head in self.heads:
        if 'temperature' in head.__dict__.keys():
          head.temperature *= temperature_update_scale
    return

  def update_heads_eps(self, eps_update_scale):
    for head in self.heads:
        if 'eps' in head.__dict__.keys():
          head.eps *= eps_update_scale
    return

  def train(self,
            dataset,
            validation_dataset=None,
            epochs=1,
            batch_size=200,
            shuffle=True,
            **kwargs):
    losses = []
    validation_losses = []

    temperature_update_scale = kwargs.pop('temperature_update', None)
    eps_update_scale = kwargs.pop('eps_update', None)

    for epoch in range(epochs):
      if shuffle:
        dataset = dataset.shuffle(buffer_size=dataset.cardinality())

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
        validation_losses.append(-self.elbo(validation_dataset))

    return losses, validation_losses

  def get_config(self):
    config = {
        'name': self.name,
        'dim_latent': self.dim_latent,
        'beta': self.beta,
        'loss_scales': self.loss_scales.numpy().tolist(),
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
        config.pop('optimizer_config', None)
      json.dump(config, f)

    self.encoder.save_weights(f'{model_path}/encoder')
    self.first_encoder.save_weights(f'{model_path}/first_encoder')
    for i, head in enumerate(self.heads):
      head.save_weights(f'{model_path}/head{i}')

    if include_optimizer:
      with h5py.File(f'{model_path}/optimizer_state.hdf5', 'w') as h5_store:
        for i, v in enumerate(self.optimizer.variables):
          h5_store.create_dataset(name=str(i), data=v.numpy())
      
  def load(model_path, include_optimizer=False):
    with open(f'{model_path}/model_config.json', 'r') as f:
      config = json.load(f)

    if not include_optimizer:
      config.pop('optimizer_config', None)
    model = FACTMx_model.from_config(config)

    model.encoder.load_weights(f'{model_path}/encoder')
    model.first_encoder.load_weights(f'{model_path}/first_encoder')
    for i, head in enumerate(model.heads):
      head.load_weights(f'{model_path}/head{i}')

    if include_optimizer:
      with h5py.File(f'{model_path}/optimizer_state.hdf5', 'r') as h5_store:
        model.optimizer.build(model.t_vars)
        model.optimizer.load_own_variables(h5_store)

    return model
