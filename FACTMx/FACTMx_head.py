import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from logging import warning

try:
  import tensorflow_model_optimization as tfmot
  from tensorflow_model_optimization.python.core.keras.compat import keras
  _TFMOT_IS_LOADED = True
except ImportError:
  warning('TensorFlow Resources Model optimization module not found, weight pruning is disabled.')
  import tensorflow.keras as keras
  _TFMOT_IS_LOADED = False


from typing import Tuple

from FACTMx.custom_keras_layers import ConstantResponse

class FACTMx_head(tf.Module):
  dim: int
  dim_preencoded: int
  dim_latent: int
  head_name: str
  layers: dict

  def __init__(self, dim, dim_latent, head_name, dim_preencoded=None):
    self.dim = dim
    self.dim_latent = dim_latent
    self.dim_preencoded = dim_preencoded if dim_preencoded is not None else dim
    self.head_name = head_name
    self.layers = {}

  def encode(self, data):
    pass

  def decode(self, latent, data):
    pass

  def save_weights(self, head_path):
    for key, layer in self.layers.items():
      layer.save_weights(f'{head_path}_{key}.weights.h5')

  def load_weights(self, head_path):
    for key, layer in self.layers.items():
      layer.load_weights(f'{head_path}_{key}.weights.h5')

  def get_config(self):
    config = {
      'dim': self.dim,
      'dim_latent': self.dim_latent,
      'dim_preencoded': self.dim_preencoded,
      'head_name': self.head_name
    }
    return config

  def factory(head_type, **kwargs):
    FACTMx_head_map = {head.head_type: head for head in FACTMx_head.__subclasses__()}
    return FACTMx_head_map[head_type](**kwargs)


class FACTMx_head_Bernoulli(FACTMx_head):
  head_type = 'Bernoulli'

  def __init__(self,
               dim,
               dim_latent,
               head_name,
               eps=1E-5,
               layer_configs={'logits':'linear'},
               **kwargs):
    super().__init__(dim, dim_latent, head_name,)
    self.eps = eps

    logits_config = layer_configs.pop('logits', 'linear')
    if logits_config == 'linear':
      self.layers['logits'] = keras.Sequential(
                                               [keras.Input(shape=(self.dim_latent,)),
                                                keras.layers.Dense(self.dim)]
                                            )
    else:
      self.layers['logits'] = keras.Sequential.from_config(logits_config)

    assert self.layers['logits'].output_shape == (None, self.dim)
    assert self.layers['logits'].input_shape == (None, self.dim_latent)

    self.t_vars = self.layers['logits'].trainable_variables


  def decode_params(self, latent):
    #decode logits from a latent point
    return self.layers['logits'](latent)

  def make_decoder(self, latent):
    #return the decoding distribution given its latent point
    logits = self.decode_params(latent)
    return tfp.distributions.Bernoulli(logits=logits)

  def decode(self, latent, data):
    #decode a sample from latent
    return self.make_decoder(latent).sample()

  def encode(self, data):
    #give logits
    logits = tf.math.log((data+self.eps) / (1-data+self.eps))
    return {'encoder_input':data}

  def loss(self, data, latent, beta=1):
    #return -loglikelihood of data given its latent point
    log_prob = self.make_decoder(latent).log_prob(data)

    loss = -tf.reduce_mean(log_prob)
    loss += tf.reduce_sum(self.layers['logits'].losses)

    return loss

  def get_config(self):
    config = super().get_config()
    config.update({
        'head_type': self.head_type,
        'layer_configs': {'logits': self.layers['logits'].get_config()}
    })
    return config

  def from_config(config):
    return FACTMx_head_Bernoulli(**config)



class FACTMx_head_Multinomial(FACTMx_head):
  head_type = 'Multinomial'

  def __init__(self,
               dim_pos,
               dim_cat,
               dim, dim_latent, head_name,
               layer_configs={'logits':'linear'},
               eps = 1E-3,
               **kwargs):
    #dim is the dimension of head's output to encoder
    super().__init__(dim, dim_latent, head_name)
    self.eps = eps
    self.dim_pos = dim_pos
    self.dim_cat = dim_cat
    _dim_logits = dim_pos * dim_cat

    logits_config = layer_configs.pop('logits', 'linear')
    if logits_config == 'linear':
      self.layers['logits'] = tf.keras.Sequential(
                              [tf.keras.Input(shape=(self.dim_latent,)),
                              tf.keras.layers.Dense(_dim_logits)]
                      )
    else:
      self.layers['logits'] = tf.keras.Sequential.from_config(logits_config)

    assert self.layers['logits'].output_shape == (None, _dim_logits)
    assert self.layers['logits'].input_shape == (None, self.dim_latent)

    self.t_vars = self.layers['logits'].trainable_variables

    preencoder_config = layer_configs.pop('preencoder', None)
    if preencoder_config is None:
      self.preencoder = False
    else:
      self.preencoder = True
      self.layers['preencoder'] = tf.keras.Sequential.from_config(preencoder_config)
      self.t_vars += self.layers['preencoder'].trainable_variables

      assert self.layers['preencoder'].output_shape == (None, self.dim)
      assert self.layers['preencoder'].input_shape == (None, dim_pos * dim_cat)


  def decode_params(self, latent):
    #decode logits from a latent point
    return tf.reshape(self.layers['logits'](latent), shape=(-1, self.dim_pos, self.dim_cat))

  def make_decoder(self, latent, counts):
    #return the decoding distribution given its latent point
    logits = self.decode_params(latent)
    return tfp.distributions.Multinomial(total_count=counts, logits=logits)

  def decode(self, latent, data):
    #decode a sample from latent
    observations, counts = data
    return self.make_decoder(latent, counts).sample()

  def encode(self, data):
    #give logits or preencoding to encode
    observations, counts = data
    if self.preencoder:
      preencoder_input = tf.reshape(observations, shape=(-1, self.dim_pos * self.dim_cat))
      encoder_input = self.layers['preencoder'](preencoder_input)
      return {'encoder_input': encoder_input}
    else:
      encoder_input = tf.reshape(observations, shape=(-1, self.dim_pos * self.dim_cat))
      return {'encoder_input': encoder_input}

  def loss(self, data, latent, beta=1):
    #return -loglikelihood of data given its latent point
    observations, counts = data
    log_prob = self.make_decoder(latent, counts).log_prob(observations)

    loss = -tf.reduce_sum(log_prob) / observations.shape[0]
    loss += tf.reduce_sum(self.layers['logits'].losses)

    return loss

  def get_config(self):
    config = {
        'head_type': self.head_type,
        'dim_pos': self.dim_pos,
        'dim_cat': self.dim_cat,
        'dim': self.dim,
        'dim_latent': self.dim_latent,
        'head_name': self.head_name,
        'layer_configs': {key: layer.get_config() for key, layer in self.layers.items()},
        'eps': self.eps,
    }
    return config

  def from_config(config):
    return FACTMx_head_Multinomial(**config)


class FACTMx_head_MultiNormal(FACTMx_head):
  head_type = 'MultiNormal'

  def __init__(self,
               dim, dim_latent, head_name,
               layer_configs={'loc':'linear', 'scale':'linear'},
               independent=False,
               eps=1E-3, 
               **kwargs):
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
                                                activation='sigmoid',
                                                bias_initializer={'class_name':'Constant', 'config':{'value':np.log(eps)}}),
                               tf.keras.layers.Rescaling(scale=.5)])
    else:
      self.layers['scale'] = tf.keras.Sequential.from_config(scale_config)

    self.t_vars = tuple(var for layer in self.layers.values() for var in layer.trainable_variables)

  def decode_params(self, latent):
    #decode loc and cov from a latent point
    loc = self.layers['loc'](latent)
    scale_diag = self.layers['scale'](latent) + self.eps

    return loc, scale_diag

  def make_decoder(self, latent):
    #return the decoding distribution given its latent point
    loc, scale = self.decode_params(latent)
    return tfp.distributions.MultivariateNormalDiag(loc, scale)

  def encode(self, data):
    #no pass needed
    return {'encoder_input':data}

  def decode(self, latent, data):
    #decode a sample from latent
    return self.make_decoder(latent).sample()

  def loss(self, data, latent, beta=1):
    #return -loglikelihood of data given its latent point and any additional 
    loc, scale = self.decode_params(latent)
    log_prob = tfp.distributions.MultivariateNormalDiag(loc, scale).log_prob(data)
    
    loss = -tf.reduce_mean(log_prob)
    loss += tf.reduce_mean(np.log(scale)) * 1E3
    for layer in self.layers.values():
      loss += tf.reduce_sum(layer.losses)

    return loss

  def get_config(self):
    config = super().get_config()
    config.update({
                "head_type": self.head_type,
                "eps": self.eps,
                "layer_configs": {key: layer.get_config() for key, layer in self.layers.items()}
             })
    return config

  def from_config(config):
    return FACTMx_head_MultiNormal(**config)

  
