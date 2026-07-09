"""
Encoder modules for FACTMx latent-variable models.

An encoder receives the concatenated per-head representations produced by
FACTMx heads and returns a TensorFlow Probability distribution over latent
coordinates.  Concrete encoders implement different aggregation strategies:
learned linear maps, attention over equally-sized head embeddings, or a simple
mean over equally-sized head embeddings.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras

from typing import Tuple, Dict

from FACTMx.custom_keras_layers import ConstantResponse

def all_equal(list):
  """Return True when every element of the provided list has the same value."""
  return len(set(list)) == 1



class FACTMx_encoder(tf.Module):
  """
  Base class for FACTMx encoders.
  
  Concrete subclasses must expose ``layers`` and ``t_vars`` and implement the
  encoding distribution used by the top-level model.
  """
  head_dims: Tuple[int]
  dim_latent: int

  def __init__(self, dim_latent, head_dims, name=None,):
    """
    Store shared encoder metadata.
    
    Args:
      dim_latent: Dimensionality of the shared latent representation.
      head_dims: Output dimensions contributed by each data head to the encoder.
      name: Optional TensorFlow module name.
    """
    super().__init__(name=name)
    self.dim_latent = dim_latent
    self.head_dims = head_dims

  def save_weights(self, encoder_path):
    """Save each registered Keras sub-layer to a separate weight file."""
    for key, layer in self.layers.items():
      layer.save_weights(f'{encoder_path}_{key}.weights.h5')

  def load_weights(self, encoder_path):
    """Load each registered Keras sub-layer from a separate weight file."""
    for key, layer in self.layers.items():
      layer.load_weights(f'{encoder_path}_{key}.weights.h5')

  def factory(encoder_type='Linear', **kwargs):
    """Instantiate an encoder subclass by its ``encoder_type`` string."""
    FACTMx_encoder_map = {encoder.encoder_type: encoder for encoder in FACTMx_encoder.__subclasses__()}
    return FACTMx_encoder_map[encoder_type](**kwargs)




class FACTMx_encoder_Linear(FACTMx_encoder):
  """
  Linear variational encoder over concatenated head embeddings.
  
  The encoder uses one Keras network to produce the posterior mean and a second
  network to produce a diagonal Cholesky factor for a multivariate Normal
  posterior.
  """
  encoder_type = 'Linear'
  
  def __init__(self, dim_latent, head_dims,
               layer_configs={'loc':'linear', 'scale':'linear'},
               name=None, 
               prior_params=None, 
               eps=1E-5,):
    """
    Build the linear encoder networks and prior distribution.
    
    Args:
      dim_latent: Dimensionality of latent samples.
      head_dims: Dimensions of concatenated head encodings.
      layer_configs: Keras configs or ``'linear'`` sentinels for loc and scale.
      name: Optional TensorFlow module name.
      prior_params: Optional parameters for ``MultivariateNormalTriL`` prior.
      eps: Small positive value added to scales for numerical stability.
    """
    super().__init__(dim_latent, head_dims, name)
    self.eps = eps
    self.layers = {}

    # Mean and scale networks may be supplied as serialized Keras configs;
    # 'linear' keeps backward-compatible defaults for older saved models.
    loc_config = layer_configs.pop('loc', 'linear')
    if loc_config == 'linear':
      self.layers['loc'] = keras.Sequential(
                              [tf.keras.Input(shape=(sum(head_dims),)),
                               tf.keras.layers.Dense(units=dim_latent,
                                                     kernel_initializer='orthogonal')]
      )
    else:
      self.layers['loc'] = keras.Sequential.from_config(loc_config)

    scale_config = layer_configs.pop('scale', 'linear')
    if scale_config == 'linear':
      self.layers['scale'] = tf.keras.Sequential(
                              [tf.keras.Input(shape=(sum(head_dims),)),
                               ConstantResponse(units=dim_latent,
                                                activation='relu',
                                                bias_initializer={'class_name':'Constant', 'config':{'value':eps}})]
      )
    else:
      self.layers['scale'] = tf.keras.Sequential.from_config(scale_config)

    self.t_vars = tuple(var for layer in self.layers.values() for var in layer.trainable_variables)

    if prior_params is None:
      loc = tf.zeros(dim_latent)
      scale_tril = tf.eye(dim_latent)
      self.prior = tfp.distributions.MultivariateNormalTriL(loc, scale_tril)
    else:
      self.prior = tfp.distributions.MultivariateNormalTriL(**prior_params)


  def encode_params(self, data):
    """Return posterior mean and Cholesky diagonal for a batch of encoder inputs."""
    loc = self.layers['loc'](data)
    scale_diag = self.layers['scale'](data) + self.eps
    scale_diag = tf.linalg.diag(scale_diag)

    return loc, scale_diag

  def make_encoder(self, data):
    """Construct the TensorFlow Probability posterior distribution for data."""
    loc, scale_tril = self.encode_params(data)
    return tfp.distributions.MultivariateNormalTriL(loc, scale_tril)

  def encode(self, data, deterministic=False):
    """Sample latent points, or return posterior means when deterministic=True."""
    if deterministic:
      loc, scale = self.encode_params(data)
      return loc
    else:
      return self.make_encoder(data).sample()

  def encode_with_loss(self, data):
    """Sample latent points and return the encoder KL regularisation loss."""
    encoder = self.make_encoder(data)

    sample = encoder.sample()
    loss = tf.reduce_mean(encoder.kl_divergence(self.prior))

    for layer in self.layers.values():
      loss += tf.reduce_sum(layer.losses)

    return sample, loss

  def loss(self, data):
    """Return only the KL loss between the posterior and configured prior."""
    encoder = self.make_encoder(data)
    loss = tf.reduce_mean(encoder.kl_divergence(self.prior))

    for layer in self.layers.values():
      loss += tf.reduce_sum(layer.losses)

    return loss

  def get_config(self):
    """Serialize the encoder architecture, prior, and Keras layer configs."""
    config = {  
                "encoder_type": self.encoder_type,
                "dim_latent": self.dim_latent,
                "head_dims": self.head_dims,
                "eps": self.eps,
                "prior_params": {'loc':self.prior.loc.numpy().tolist(),
                                 'scale_tril':self.prior.scale_tril.numpy().tolist()},
                "layer_configs": {key: layer.get_config() for key, layer in self.layers.items()}
             }

    return config

  def from_config(config):
    """Reconstruct a linear encoder from a serialized config dictionary."""
    return FACTMx_encoder_Linear(**config)



class FACTMx_encoder_Attention(FACTMx_encoder):
  """
  Attention-based encoder for equally-sized head embeddings.
  
  Each head representation is treated as an item in a short sequence.  Attention
  aggregates the sequence, and the variance across attended values defines a
  diagonal posterior scale.
  """
  encoder_type = 'Attention'

  def __init__(self, dim_latent, head_dims,
               name=None,
               prior_params=None,
               eps=1E-5,):
    """Initialise attention machinery and the latent prior."""
    super().__init__(dim_latent, head_dims, name)
    self.eps = eps
    self.layers = {}

    assert dim_latent == head_dims[0]
    assert all_equal(head_dims)

    # Keras Attention expects sequence-shaped tensors; encode_params reshapes
    # the concatenated head vectors into a short sequence before calling it.
    self.attention_mechanism = tf.keras.layers.Attention()
    self.layers['key_transform'] = keras.Sequential(
                              [tf.keras.Input(shape=(dim_latent,)),
                               tf.keras.layers.Dense(units=dim_latent,
                                                     use_bias=False)]
      )

    self.t_vars = tuple(var for layer in self.layers.values() for var in layer.trainable_variables)

    if prior_params is None:
      loc = tf.zeros(dim_latent)
      scale_tril = tf.eye(dim_latent)
      self.prior = tfp.distributions.MultivariateNormalTriL(loc, scale_tril)
    else:
      self.prior = tfp.distributions.MultivariateNormalTriL(**prior_params)


  def encode_params(self, data):
    """Aggregate head embeddings with attention and return posterior parameters."""
    n_heads = len(self.head_dims)
    
    #data comes concatenated to shape (n_batch, n_heads*dim_latent)
    flat_data = tf.reshape(data, shape=(-1, self.dim_latent))

    keys = self.layers['key_transform'](flat_data)
    keys = tf.reshape(keys, shape=(-1, n_heads, self.dim_latent))

    broad_data = tf.reshape(data, shape=(-1, n_heads, self.dim_latent))
    values = self.attention_mechanism([keys, broad_data])

    loc = tf.reduce_mean(values, axis=1)

    scale_diag = tf.math.reduce_variance(values, axis=1) + self.eps
    scale_diag = tf.linalg.diag(scale_diag)

    return loc, scale_diag

  def make_encoder(self, data):
    """Build the attention encoder's multivariate Normal posterior."""
    loc, scale_tril = self.encode_params(data)
    return tfp.distributions.MultivariateNormalTriL(loc, scale_tril)

  def encode(self, data):
    """Sample a latent point from the attention-based posterior."""
    return self.make_encoder(data).sample()

  def encode_with_loss(self, data):
    """Sample a latent point and compute the posterior-prior KL loss."""
    encoder = self.make_encoder(data)

    sample = encoder.sample()
    loss = tf.reduce_mean(encoder.kl_divergence(self.prior))

    for layer in self.layers.values():
      loss += tf.reduce_sum(layer.losses)

    return sample, loss

  def loss(self, data):
    """Compute the posterior-prior KL loss without returning a sample."""
    encoder = self.make_encoder(data)
    loss = tf.reduce_mean(encoder.kl_divergence(self.prior))

    for layer in self.layers.values():
      loss += tf.reduce_sum(layer.losses)

    return loss

  def get_config(self):
    """Serialize the attention encoder configuration and prior parameters."""
    config = {  
                "encoder_type": self.encoder_type,
                "dim_latent": self.dim_latent,
                "head_dims": self.head_dims,
                "eps": self.eps,
                "prior_params": {'loc':self.prior.loc.numpy().tolist(),
                                 'scale_tril':self.prior.scale_tril.numpy().tolist()},
             }

    return config

  def from_config(config):
    """Reconstruct an attention encoder from a config dictionary."""
    return FACTMx_encoder_Attention(**config)



class FACTMx_encoder_Mean(FACTMx_encoder):
  """
  Mean-pooling encoder for equally-sized head embeddings.
  
  The posterior mean is the average of the head encodings.  A learned diagonal
  scale parameter controls posterior uncertainty.
  """
  encoder_type = 'Mean'

  def __init__(self, dim_latent, head_dims,
               log_scale_diag=None,
               name=None,
               prior_params=None,
               eps=1E-5,):
    """Initialise mean-pooling encoder state and its prior distribution."""
    super().__init__(dim_latent, head_dims, name)
    self.eps = eps
    self.layers = {}

    assert dim_latent == head_dims[0]
    assert all_equal(head_dims)

    # Mean-pooling has no neural scale branch, so posterior uncertainty is a
    # learned global diagonal parameter.
    if log_scale_diag is None:
      log_scale_diag = tf.keras.initializers.Zeros()((dim_latent,))
    self.log_scale_diag = tf.keras.Variable(log_scale_diag)

    self.t_vars = (self.log_scale_diag,)

    if prior_params is None:
      loc = tf.zeros(dim_latent)
      scale_tril = tf.eye(dim_latent)
      self.prior = tfp.distributions.MultivariateNormalTriL(loc, scale_tril)
    else:
      self.prior = tfp.distributions.MultivariateNormalTriL(**prior_params)


  def encode_params(self, data):
    """Return mean-pooled posterior location and learned diagonal scale."""
    n_heads = len(self.head_dims)
    
    #data comes concatenated to shape (n_batch, n_heads*dim_latent)
    broad_data = tf.reshape(data, shape=(-1, n_heads, self.dim_latent))

    loc = tf.reduce_mean(broad_data, axis=1)

    scale_diag = tf.math.exp(self.log_scale_diag) + self.eps
    scale_diag = tf.linalg.diag(scale_diag)

    return loc, scale_diag

  def make_encoder(self, data):
    """Construct the mean encoder's multivariate Normal posterior."""
    loc, scale_tril = self.encode_params(data)
    return tfp.distributions.MultivariateNormalTriL(loc, scale_tril)

  def encode(self, data):
    """Sample a latent point from the mean-pooling posterior."""
    return self.make_encoder(data).sample()

  def encode_with_loss(self, data):
    """Sample a latent point and compute KL loss against the prior."""
    encoder = self.make_encoder(data)

    sample = encoder.sample()
    loss = tf.reduce_mean(encoder.kl_divergence(self.prior))

    for layer in self.layers.values():
      loss += tf.reduce_sum(layer.losses)

    return sample, loss

  def loss(self, data):
    """Compute only the posterior-prior KL loss."""
    encoder = self.make_encoder(data)
    loss = tf.reduce_mean(encoder.kl_divergence(self.prior))

    for layer in self.layers.values():
      loss += tf.reduce_sum(layer.losses)

    return loss

  def get_config(self):
    """Serialize mean encoder parameters, including learned log scale."""
    config = {  
                "encoder_type": self.encoder_type,
                "dim_latent": self.dim_latent,
                "head_dims": self.head_dims,
                "eps": self.eps,
                "log_scale_diag": self.log_scale_diag.numpy().tolist(),
                "prior_params": {'loc':self.prior.loc.numpy().tolist(),
                                 'scale_tril':self.prior.scale_tril.numpy().tolist()},
             }

    return config

  def from_config(config):
    """Reconstruct a mean encoder from a config dictionary."""
    return FACTMx_encoder_Mean(**config)
