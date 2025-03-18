import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from typing import Tuple

from FACTMx.custom_keras_layers import ConstantResponse

class FACTMx_head(tf.Module):
  dim: int
  dim_to_encoder: int
  dim_latent: int
  head_name: str

  def __init__(self, dim, dim_preencoded=None, dim_latent, head_name):
    self.dim = dim
    self.dim_preencoded = dim_data if dim_preencoded is None else dim_preencoded
    self.dim_latent = dim_latent
    self.head_name = head_name

  def encode(self, data):
    pass

  def decode(self, latent, data):
    pass

  def factory(head_type, **kwargs):
    FACTMx_head_map = {head.head_type: head for head in FACTMx_head.__subclasses__()}
    return FACTMx_head_map[head_type](**kwargs)


class FACTMx_head_Bernoulli(FACTMx_head):
  head_type = 'Bernoulli'

  def __init__(self,
               dim,
               dim_preencoded=None,
               dim_latent,
               head_name,
               eps=1E-5,
               decode_config='linear',
               encode_config=None,
               **kwargs):
    super().__init__(dim, dim_preencoded, dim_latent, head_name,)
    self.eps = eps

    if decode_config == 'linear':
      self.decode_model = tf.keras.Sequential(
                            [tf.keras.Input(shape=(self.dim_latent,)),
                             tf.keras.layers.Dense(self.dim)]
                          )
    else:
      self.decode_model = tf.keras.Sequential.from_config(decode_config)

    assert self.decode_model.output_shape == (None, self.dim)
    assert self.decode_model.input_shape == (None, self.dim_latent)

    if encode_config is not None:
      self.encode_model = tf.keras.Sequential.from_config(encode_config)
      assert self.encode_model.output_shape == (None, self.dim_preencoded)
      assert self.encode_model.input_shape == (None, self.dim)
    else:
      self.encode_model = None

    if self.encode_model is None:
      self.t_vars = self.decode_model.trainable_variables
    else:
      self.t_vars = [*self.decode_model.trainable_variables,
                     *self.encode_model.trainable_variables]


  def decode_params(self, latent):
    #decode logits from a latent point
    return self.decode_model(latent)

  def make_decoder(self, latent):
    #return the decoding distribution given its latent point
    logits = self.decode_params(latent)
    return tfp.distributions.Bernoulli(logits=logits)

  def decode(self, latent, data):
    #decode a sample from latent
    return self.make_decoder(latent).sample()

  def encode(self, data):
    #give logits if no encoder model is present
    if self.encode_model is None:
      preencoded = tf.math.log((data+self.eps) / (1-data+self.eps))
    else:
      preencoded = self.encode_model(data)
    return {'encoder_input':preencoded}

  def loss(self, data, latent, beta=1):
    #return -loglikelihood of data given its latent point
    log_prob = self.make_decoder(latent).log_prob(data)

    loss = -tf.reduce_mean(log_prob)
    loss += tf.reduce_sum(self.decode_model.losses)

    return loss

  def get_config(self):
    config = {
        'head_type': self.head_type,
        'dim': self.dim,
        'dim_preencoded': self.dim_preencoded,
        'dim_latent': self.dim_latent,
        'head_name': self.head_name,
        'decode_config': self.decode_model.get_config(),
        'encode_config': self.encode_model.get_config() if self.encode_model is not None else None
    }
    return config

  def from_config(config):
    return FACTMx_head_Bernoulli(**config)

  def save_weights(self, head_path):
    self.decode_model.save_weights(f'{head_path}_decode.weights.h5')
    if self.encode_model is not None:
      self.encode_model.save_weights(f'{head_path}_encode.weights.h5')

  def load_weights(self, head_path):
    self.decode_model.load_weights(f'{head_path}_decode.weights.h5')
    if self.encode_model is not None:
      self.encode_model.load_weights(f'{head_path}_encode.weights.h5')


class FACTMx_head_Multinomial(FACTMx_head):
  head_type = 'Multinomial'

  def __init__(self,
               dim, 
               dim_preencoded=None, 
               dim_latent, head_name,
               decode_config='linear',
               eps = 1E-1, **kwargs):
    super().__init__(dim, dim_preencoded, dim_latent, head_name)
    self.eps = eps

    if decode_config == 'linear':
      self.decode_model = tf.keras.Sequential(
                            [tf.keras.Input(shape=(self.dim_latent,)),
                             tf.keras.layers.Dense(self.dim)]
                          )
    else:
      self.decode_model = tf.keras.Sequential.from_config(decode_config)

    assert self.decode_model.output_shape == (None, self.dim)
    assert self.decode_model.input_shape == (None, self.dim_latent)

    if encode_config is not None:
      self.encode_model = tf.keras.Sequential.from_config(encode_config)
      assert self.encode_model.output_shape == (None, self.dim_preencoded)
      assert self.encode_model.input_shape == (None, self.dim)
    else:
      self.encode_model = None

    if self.encode_model is None:
      self.t_vars = self.decode_model.trainable_variables
    else:
      self.t_vars = [*self.decode_model.trainable_variables,
                     *self.encode_model.trainable_variables]


  def decode_params(self, latent):
    #decode logits from a latent point
    return self.decode_model(latent)

  def make_decoder(self, latent, counts):
    #return the decoding distribution given its latent point
    logits = self.decode_params(latent)
    padded_logits = tf.pad(logits,
                           tf.constant([[0, 0], [1, 0]]),
                           'CONSTANT')
    return tfp.distributions.Multinomial(total_count=counts, logits=padded_logits)

  def decode(self, latent, data):
    #decode a sample from latent
    counts = data[1]
    return self.make_decoder(latent, counts).sample()

  def encode(self, data):
    #give logits to encode
    logits = tf.math.log(data[0] + self.eps)
    normalized = logits[:,1:] - tf.reshape(logits[:,0], shape=(-1,1))
    return {'encoder_input': normalized}

  def loss(self, data, latent, beta=1):
    #return -loglikelihood of data given its latent point
    observations, counts = data
    log_prob = self.make_decoder(latent, counts).log_prob(observations)

    loss = -tf.reduce_mean(log_prob)
    loss += tf.reduce_sum(self.decode_model.losses)

    return loss 

  def get_config(self):
    config = {
        'head_type': self.head_type,
        'dim': self.dim,
        'dim_latent': self.dim_latent,
        'head_name': self.head_name,
        'decode_config': self.decode_model.get_config()
    }
    return config

  def from_config(config):
    return FACTMx_head_Multinomial(**config)

  def save_weights(self, head_path):
    self.decode_model.save_weights(f'{head_path}.weights.h5')

  def load_weights(self, head_path):
    self.decode_model.load_weights(f'{head_path}.weights.h5')


class FACTMx_head_MultiNormal(FACTMx_head):
  head_type = 'MultiNormal'

  def __init__(self,
               dim, dim_latent, head_name,
               layer_configs={'loc':'linear', 'scale':'linear'},
               eps=1E-2, **kwargs):
    super().__init__(dim, dim_latent, head_name)
    self.eps = eps
    self.layers = {}

    loc_config = layer_configs.pop('loc', 'linear')
    if loc_config == 'linear':
      self.layers['loc'] = tf.keras.Sequential(
                              [tf.keras.Input(shape=(self.dim_latent,)),
                               tf.keras.layers.Dense(units=self.dim,
                                                     kernel_initializer='orthogonal')]
                           )
    else:
      self.layers['loc'] = tf.keras.Sequential.from_config(loc_config)

    scale_config = layer_configs.pop('scale', 'linear')
    if scale_config == 'linear':
      self.layers['scale'] = tf.keras.Sequential(
                              [tf.keras.Input(shape=(self.dim_latent,)),
                               ConstantResponse(units=self.dim,
                                                activation='relu',
                                                bias_initializer='zeros')]
                             )
    else:
      self.layers['scale'] = tf.keras.Sequential.from_config(scale_config)

    self.t_vars = tuple(var for layer in self.layers.values() for var in layer.trainable_variables)


  def decode_params(self, latent):
    #decode loc and cov from a latent point
    loc = self.layers['loc'](latent)

    scale_diag = self.layers['scale'](latent) + self.eps
    scale_diag = tf.linalg.diag(scale_diag)

    return loc, scale_diag

  def make_decoder(self, latent):
    #return the decoding distribution given its latent point
    loc, scale_tril = self.decode_params(latent)
    return tfp.distributions.MultivariateNormalTriL(loc, scale_tril)

  def encode(self, data):
    #no pass needed
    return {'encoder_input':data}

  def decode(self, latent, data):
    #decode a sample from latent
    return self.make_decoder(latent).sample()

  def loss(self, data, latent, beta=1):
    #return -loglikelihood of data given its latent point and any additional losses
    log_prob = self.make_decoder(latent).log_prob(data)
    
    loss = -tf.reduce_mean(log_prob)
    for layer in self.layers.values():
      loss += tf.reduce_sum(layer.losses)

    return loss

  def get_config(self):
    config = {
                "head_type": self.head_type,
                "dim": self.dim,
                "dim_latent": self.dim_latent,
                "head_name": self.head_name,
                "eps": self.eps,
                "layer_configs": {key: layer.get_config() for key, layer in self.layers.items()}
             }
    return config

  def from_config(config):
    return FACTMx_head_MultiNormal(**config)

  def save_weights(self, head_path):
    for key, layer in self.layers.items():
      layer.save_weights(f'{head_path}_{key}.weights.h5')

  def load_weights(self, head_path):
    for key, layer in self.layers.items():
      layer.load_weights(f'{head_path}_{key}.weights.h5')
