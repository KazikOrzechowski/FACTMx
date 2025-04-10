import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

try:
  import tensorflow_model_optimization as tfmot
  from tensorflow_model_optimization.python.core.keras.compat import keras
  _TFMOT_IS_LOADED = True
except ImportError:
  warning('TensorFlow Resources Model optimization module not found, weight pruning is disabled.')
  import tensorflow.keras as keras
  _TFMOT_IS_LOADED = False

from typing import Tuple, Dict

from FACTMx.custom_keras_layers import ConstantResponse

class FACTMx_encoder(tf.Module):
  head_dims: Tuple[int]
  dim_latent: int
  eps: float

  def __init__(self, dim_latent, head_dims,
               layer_configs={'loc':'linear', 'scale':'linear'},
               name=None, 
               prior_params=None, 
               eps=1E-5,
               pruning_params={'loc':None, 'scale':None}):
    super().__init__(name=name)
    self.pruning_params = pruning_params
    self.dim_latent = dim_latent
    self.head_dims = head_dims
    self.eps = eps
    self.layers = {}

    loc_config = layer_configs.pop('loc', 'linear')
    if loc_config == 'linear':
      self.layers['loc'] = keras.Sequential(
                              [tf.keras.Input(shape=(sum(head_dims),)),
                               tf.keras.layers.Dense(units=dim_latent,
                                                     kernel_initializer='orthogonal')]
      )
    else:
      self.layers['loc'] = keras.Sequential.from_config(loc_config)

    loc_pruning = pruning_params.pop('loc', None)
    if loc_pruning is not None and _TFMOT_IS_LOADED:
      self.layers['loc'] = tfmot.sparsity.keras.prune_low_magnitude(self.layers['loc'], **loc_pruning)

    scale_config = layer_configs.pop('scale', 'linear')
    if scale_config == 'linear':
      self.layers['scale'] = tf.keras.Sequential(
                              [tf.keras.Input(shape=(sum(head_dims),)),
                               ConstantResponse(units=dim_latent,
                                                activation='relu',
                                                bias_initializer='zeros')]
      )
    else:
      self.layers['scale'] = tf.keras.Sequential.from_config(scale_config)

    scale_pruning = pruning_params.pop('scale', None)
    if scale_pruning is not None and _TFMOT_IS_LOADED:
      self.layers['scale'] = tfmot.sparsity.keras.prune_low_magnitude(self.layers['scale'], **scale_pruning)

    self.t_vars = tuple(var for layer in self.layers.values() for var in layer.trainable_variables)

    if prior_params is None:
      loc = tf.zeros(dim_latent)
      scale_tril = tf.eye(dim_latent)
      self.prior = tfp.distributions.MultivariateNormalTriL(loc, scale_tril)
    else:
      self.prior = tfp.distributions.MultivariateNormalTriL(**prior_params)


  def encode_params(self, data):
    #return loc and cov parameters of the latent distributions for data points
    loc = self.layers['loc'](data)
    scale_diag = self.layers['scale'](data) + self.eps
    scale_diag = tf.linalg.diag(scale_diag)

    return loc, scale_diag

  def make_encoder(self, data):
    #return an encoding distribution for data points
    loc, scale_tril = self.encode_params(data)
    return tfp.distributions.MultivariateNormalTriL(loc, scale_tril)

  def encode(self, data):
    #encode data to latent points
    return self.make_encoder(data).sample()

  def encode_with_loss(self, data):
    encoder = self.make_encoder(data)

    sample = encoder.sample()
    loss = tf.reduce_mean(encoder.kl_divergence(self.prior))

    for layer in self.layers.values():
      loss += tf.reduce_sum(layer.losses)

    return sample, loss

  def loss(self, data):
    #return kl divergence loss between the encoded posteriors and the prior distribution
    encoder = self.make_encoder(data)
    loss = tf.reduce_mean(encoder.kl_divergence(self.prior))

    for layer in self.layers.values():
      loss += tf.reduce_sum(layer.losses)

    return loss

  def get_config(self):
    config = {
                "dim_latent": self.dim_latent,
                "head_dims": self.head_dims,
                "eps": self.eps,
                "prior_params": {'loc':self.prior.loc.numpy().tolist(),
                                 'scale_tril':self.prior.scale_tril.numpy().tolist()},
                "pruning_params": self.pruning_params,
                "layer_configs": {key: layer.get_config() for key, layer in self.layers.items()}
             }

    return config

  def from_config(config):
    return FACTMx_encoder(**config)

  def save_weights(self, encoder_path):
    for key, layer in self.layers.items():
      layer.save_weights(f'{encoder_path}_{key}.weights.h5')

  def load_weights(self, encoder_path):
    for key, layer in self.layers.items():
      layer.load_weights(f'{encoder_path}_{key}.weights.h5')

  def save_weights(self, encoder_path):
    for key, layer in self.layers.items():
      layer.save_weights(f'{encoder_path}_{key}.weights.h5')

  def load_weights(self, encoder_path):
    for key, layer in self.layers.items():
      layer.load_weights(f'{encoder_path}_{key}.weights.h5')
