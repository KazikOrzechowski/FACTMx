import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from FACTMx.FACTMx_head import FACTMx_head
from FACTMx.custom_keras_layers import ConstantResponse

class FACTMx_head_ZeroInflatedMultiNormal(FACTMx_head):
  head_type = 'ZeroInflatedMultiNormal'

  def __init__(self,
               dim, dim_latent, head_name,
               layer_configs={'loc':'linear', 'scale':'linear'},
               presigmoid_zero_probs=None,
               zero_temperature=1E-3,
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

    if presigmoid_zero_probs is None:
      presigmoid_zero_probs = tf.keras.initializers.Zeros()(shape=(self.dim,))
    self.presigmoid_zero_probs = tf.keras.Variable(presigmoid_zero_probs,
                                                   shape=(1, self.dim),
                                                   trainable=True,
                                                   dtype=tf.float32)

    self.zero_temperature = zero_temperature
    self.zero_component = tfp.distributions.MultivariateNormalDiag(tf.zeros((1, self.dim, 1)),
                                                                   self.zero_temperature*tf.ones((1, self.dim, 1)))

    self.t_vars = (*[var for layer in self.layers.values() for var in layer.trainable_variables],
                   self.presigmoid_zero_probs)


  def decode_params(self, latent):
    #decode loc and cov from a latent point
    loc = self.layers['loc'](latent)
    loc = tf.reshape(loc, (-1, self.dim, 1))

    scale_diag = self.layers['scale'](latent) + self.eps
    scale_diag = tf.reshape(scale_diag, (-1, self.dim, 1))

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
    #return -loglikelihood of data given its latent point and any additional losses
    data_1d = tf.reshape(data, (-1, self.dim, 1))

    log_prob_normal_component = self.make_decoder(latent).log_prob(data_1d)
    log_prob_zero_component = self.zero_component.log_prob(data_1d)

    zero_probs = tf.math.sigmoid(self.presigmoid_zero_probs)

    log_zero_probs = tf.math.log(zero_probs)
    log_zero_probs = tf.reshape(log_zero_probs, shape=(1, -1))

    log_normal_probs = tf.math.log(1-zero_probs)
    log_normal_probs = tf.reshape(log_normal_probs, shape=(1, -1))

    stacked = tf.stack([log_zero_probs + log_prob_zero_component,
                        log_normal_probs + log_prob_normal_component], axis=2)

    log_like_total = tf.math.reduce_logsumexp(stacked, axis=2)

    loss = -tf.reduce_sum(log_like_total)
    batch_size = data.shape[0]
    loss /= batch_size
    for layer in self.layers.values():
      loss += tf.reduce_sum(layer.losses)

    return loss

  def get_config(self):
    config = {
                "head_type": self.head_type,
                "dim": self.dim,
                "dim_latent": self.dim_latent,
                "head_name": self.head_name,
                "presigmoid_zero_probs": self.presigmoid_zero_probs.numpy().tolist(),
                "zero_temperature": self.zero_temperature,
                "eps": self.eps,
                "layer_configs": {key: layer.get_config() for key, layer in self.layers.items()}
             }
    return config

  def from_config(config):
    return FACTMx_head_ZeroInflatedMultiNormal(**config)

  def save_weights(self, head_path):
    for key, layer in self.layers.items():
      layer.save_weights(f'{head_path}_{key}.weights.h5')

  def load_weights(self, head_path):
    for key, layer in self.layers.items():
      layer.load_weights(f'{head_path}_{key}.weights.h5')
