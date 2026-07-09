"""
Core data heads for FACTMx.

A FACTMx head is responsible for translating one observed data modality into an
encoder input and for defining the decoder likelihood used by the ELBO.  This
module contains generic Bernoulli, Multinomial, and multivariate Normal heads.
Specialised heads live in the neighbouring GMM and Topic modules.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras


from typing import Tuple

from FACTMx.custom_keras_layers import ConstantResponse

class FACTMx_head(tf.Module):
  """
  Base class for FACTMx modality heads.
  
  A head converts raw modality data into an encoder input and defines a decoder
  likelihood for reconstructing that modality from the shared latent variable.
  """
  dim: int
  dim_preencoded: int
  dim_latent: int
  head_name: str
  layers: dict

  def __init__(self, dim, dim_latent, head_name, dim_preencoded=None):
    """Store head dimensions, latent dimension, display name, and layer registry."""
    self.dim = dim
    self.dim_latent = dim_latent
    self.dim_preencoded = dim_preencoded if dim_preencoded is not None else dim
    self.head_name = head_name
    self.layers = {}

  def encode(self, data):
    """Encode raw head data into a dictionary containing ``encoder_input``."""
    pass

  def decode(self, latent, data):
    """Decode a latent representation into samples for this head's data type."""
    pass

  def save_weights(self, head_path):
    """Save all registered head layers using a common path prefix."""
    for key, layer in self.layers.items():
      layer.save_weights(f'{head_path}_{key}.weights.h5')

  def load_weights(self, head_path):
    """Load all registered head layers using a common path prefix."""
    for key, layer in self.layers.items():
      layer.load_weights(f'{head_path}_{key}.weights.h5')

  def get_config(self):
    """Return the common head configuration fields shared by subclasses."""
    config = {
      'dim': self.dim,
      'dim_latent': self.dim_latent,
      'dim_preencoded': self.dim_preencoded,
      'head_name': self.head_name
    }
    return config

  def factory(head_type, **kwargs):
    """Instantiate a head subclass by its ``head_type`` string."""
    FACTMx_head_map = {head.head_type: head for head in FACTMx_head.__subclasses__()}
    return FACTMx_head_map[head_type](**kwargs)


class FACTMx_head_Bernoulli(FACTMx_head):
  """Bernoulli decoder head for binary observations."""
  head_type = 'Bernoulli'

  def __init__(self,
               dim,
               dim_latent,
               head_name,
               eps=1E-5,
               layer_configs={'logits':'linear'},
               **kwargs):
    """Create the latent-to-logits network for binary data."""
    super().__init__(dim, dim_latent, head_name,)
    self.eps = eps

    # A string value of 'linear' requests the package's default one-layer map;
    # otherwise callers can pass a serialized Keras Sequential config.
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
    """Return Bernoulli logits predicted from latent points."""
    return self.layers['logits'](latent)

  def make_decoder(self, latent):
    """Build the Bernoulli likelihood distribution for latent points."""
    logits = self.decode_params(latent)
    return tfp.distributions.Bernoulli(logits=logits)

  def decode(self, latent, data):
    """Sample binary observations from the Bernoulli decoder."""
    return self.make_decoder(latent).sample()

  def encode(self, data):
    """Pass binary observations directly to the shared encoder."""
    logits = tf.math.log((data+self.eps) / (1-data+self.eps))
    return {'encoder_input':data}

  def loss(self, data, latent, beta=1):
    """Compute negative mean Bernoulli log-likelihood plus layer losses."""
    log_prob = self.make_decoder(latent).log_prob(data)

    loss = -tf.reduce_mean(log_prob)
    loss += tf.reduce_sum(self.layers['logits'].losses)

    return loss

  def get_config(self):
    """Serialize Bernoulli head configuration and logits layer config."""
    config = super().get_config()
    config.update({
        'head_type': self.head_type,
        'layer_configs': {'logits': self.layers['logits'].get_config()}
    })
    return config

  def from_config(config):
    """Reconstruct a Bernoulli head from a config dictionary."""
    return FACTMx_head_Bernoulli(**config)



class FACTMx_head_Multinomial(FACTMx_head):
  """
  Multinomial decoder head for grouped categorical count observations.
  
  Data is expected as ``(observations, counts)`` where observations have position
  and category axes and counts provide total counts for each position.
  """
  head_type = 'Multinomial'

  def __init__(self,
               dim_pos,
               dim_cat,
               dim, dim_latent, head_name,
               layer_configs={'logits':'linear'},
               eps = 1E-3,
               encode_logits = True,
               **kwargs):
    """Create multinomial logits and optional preencoder networks."""
    super().__init__(dim, dim_latent, head_name)
    self.eps = eps
    self.dim_pos = dim_pos
    self.dim_cat = dim_cat
    _dim_logits = dim_pos * dim_cat
    self.encode_logits = encode_logits

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

    # Optional preencoders let callers learn a modality-specific compression
    # before concatenating this head with other encoder inputs.
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
    """Return logits reshaped to ``(batch, dim_pos, dim_cat)``."""
    logits = self.layers['logits'](latent)
    log_eps = tf.constant(tf.math.log(self.eps), shape=logits.shape)

    #minimum probs is eps
    logits = tf.reduce_logsumexp(tf.stack([logits, log_eps]), axis=0)
    return tf.reshape(logits, shape=(-1, self.dim_pos, self.dim_cat))

  def make_decoder(self, latent, counts):
    """Build a Multinomial likelihood with supplied total counts."""
    logits = self.decode_params(latent)
    return tfp.distributions.Multinomial(total_count=counts, logits=logits)

  def decode(self, latent, data):
    """Sample grouped categorical counts from the multinomial decoder."""
    observations, counts = data
    return self.make_decoder(latent, counts).sample()

  def encode(self, data):
    """Convert observed counts into the representation consumed by the encoder."""
    observations, counts = data
    if self.preencoder:
      preencoder_input = tf.reshape(observations, shape=(-1, self.dim_pos * self.dim_cat))
      encoder_input = self.layers['preencoder'](preencoder_input)
    elif self.encode_logits:
      encoder_input = observations / tf.expand_dims(counts, -1) + self.eps
      encoder_input = tf.math.log(encoder_input)
      encoder_input = tf.reshape(encoder_input, shape=(-1, self.dim_pos * self.dim_cat))
    else:
      encoder_input = tf.reshape(observations, shape=(-1, self.dim_pos * self.dim_cat))
    return {'encoder_input': encoder_input}

  def loss(self, data, latent, beta=1):
    """Compute scaled negative multinomial log-likelihood plus layer losses."""
    observations, counts = data
    log_prob = self.make_decoder(latent, counts).log_prob(observations)

    loss = -tf.reduce_sum(log_prob) / observations.shape[0]
    loss += tf.reduce_sum(self.layers['logits'].losses)

    return loss

  def get_config(self):
    """Serialize multinomial dimensions, flags, and Keras layers."""
    config = {
        'head_type': self.head_type,
        'dim_pos': self.dim_pos,
        'dim_cat': self.dim_cat,
        'dim': self.dim,
        'dim_latent': self.dim_latent,
        'head_name': self.head_name,
        'layer_configs': {key: layer.get_config() for key, layer in self.layers.items()},
        'eps': self.eps,
        'encode_logits': self.encode_logits,
    }
    return config

  def from_config(config):
    """Reconstruct a multinomial head from a config dictionary."""
    return FACTMx_head_Multinomial(**config)


class FACTMx_head_MultiNormal(FACTMx_head):
  """Multivariate Normal decoder head with diagonal covariance."""
  head_type = 'MultiNormal'

  def __init__(self,
               dim, dim_latent, head_name,
               layer_configs={'loc':'linear', 'scale':'linear'},
               eps=1E-3, 
               **kwargs):
    """Create latent-to-location and shared-scale networks."""
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
        
    # The scale branch is a ConstantResponse by default, so every sample shares
    # the same diagonal decoder variance unless a custom layer is configured.
    scale_config = layer_configs.pop('scale', 'linear')
    if scale_config == 'linear':
      self.layers['scale'] = tf.keras.Sequential(
                              [tf.keras.Input(shape=(self.dim_latent,)),
                               ConstantResponse(units=self.dim,
                                                activation='exponential',
                                                bias_initializer={'class_name':'Constant', 'config':{'value':np.log(eps)}})])
    else:
      self.layers['scale'] = tf.keras.Sequential.from_config(scale_config)

    self.t_vars = tuple(var for layer in self.layers.values() for var in layer.trainable_variables)

  def decode_params(self, latent):
    """Return Normal mean and diagonal scale predicted from latent points."""
    loc = self.layers['loc'](latent)
    scale_diag = self.layers['scale'](latent) + self.eps

    return loc, scale_diag

  def make_decoder(self, latent):
    """Build a diagonal multivariate Normal decoder distribution."""
    loc, scale = self.decode_params(latent)
    return tfp.distributions.MultivariateNormalDiag(loc, scale)

  def encode(self, data):
    """Pass continuous observations directly to the shared encoder."""
    return {'encoder_input':data}

  def decode(self, latent, data):
    """Sample continuous observations from the Normal decoder."""
    return self.make_decoder(latent).sample()

  def loss(self, data, latent, beta=1):
    """Compute negative mean Normal log-likelihood plus layer losses."""
    _batch_size, _data_dim = latent.shape
    loc, scale = self.decode_params(latent)
    log_prob = tfp.distributions.MultivariateNormalDiag(loc, scale).log_prob(data)
    
    loss = -tf.reduce_mean(log_prob)
    for layer in self.layers.values():
      loss += tf.reduce_sum(layer.losses)

    return loss

  def get_config(self):
    """Serialize Normal head configuration and Keras layer configs."""
    config = super().get_config()
    config.update({
                "head_type": self.head_type,
                "eps": self.eps,
                "layer_configs": {key: layer.get_config() for key, layer in self.layers.items()}
             })
    return config

  def from_config(config):
    """Reconstruct a Normal head from a config dictionary."""
    return FACTMx_head_MultiNormal(**config)

  
