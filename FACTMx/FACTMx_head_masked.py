import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

try:
  import tensorflow_model_optimization as tfmot
  from tensorflow_model_optimization.python.core.keras.compat import keras
  _TFMOT_IS_LOADED = True
except ImportError:
  import tensorflow.keras as keras
  _TFMOT_IS_LOADED = False

from FACTMx.FACTMx_head import FACTMx_head, FACTMx_head_MultiNormal
from FACTMx.FACTMx_head_GMM import FACTMx_head_GMM
from FACTMx.FACTMx_head_flexTopic import FACTMx_head_FlexTopicModel



class FACTMx_head_GMM_masked(FACTMx_head_GMM, FACTMx_head):
  head_type = 'GMM_masked'

  def __init__(self,
               layer_configs={'preencoder':'linear'},
               masked_token=None,
               **super_kwargs):
    super().__init__(**super_kwargs)
    self.head_type = 'GMM_masked'

    preencoder_config = layer_configs.pop('preencoder', 'linear')
    if preencoder_config == 'linear':
      self.layers['preencoder'] = tf.keras.Sequential(
                                        [tf.keras.Input(shape=(self.dim,)),
                                         tf.keras.layers.Dense(units=self.dim_latent,
                                                               kernel_initializer='orthogonal')]
                                      )
    else:
      self.layers['preencoder'] = tf.keras.Sequential.from_config(preencoder_config)

    if masked_token is None:
      masked_token = tf.random.normal(shape=(self.dim_latent,), stddev=.1)
    self.masked_token = tf.keras.Variable(masked_token)

    # get training variables
    self.t_vars = [*self.t_vars,
                   *self.layers['preencoder'].trainable_variables,
                   self.masked_token]


  def decode(self, latent, data):
    observed, mask = data

    return super().decode(latent, observed)


  def loss(self,
           data,
           latent,
           encoder_assignment_sample,
           encoder_assignment_logits,
           beta=1):
    observed, mask = data

    super_loss = super().loss(observed,
                              latent,
                              encoder_assignment_sample,
                              encoder_assignment_logits,
                              beta)

    return tf.reduce_sum([super_loss, *self.layers['preencoder'].losses])


  def encode(self, data):
    observed, mask = data

    encoder_dict = super().encode(observed)

    preencoded_input = encoder_dict['encoder_input']
    preencoded_input = self.layers['preencoder'](preencoded_input)
    masked_preencoded_input = (1-mask) * preencoded_input +\
                                mask * tf.expand_dims(self.masked_token, 0)

    encoder_dict['encoder_input'] = masked_preencoded_input

    return encoder_dict

  def get_config(self):
    config = super().get_config()
    config.update({
                'layer_configs': {key: layer.get_config() for key, layer in self.layers.items()},
                'masked_token': self.masked_token.numpy().tolist(),
             })
    return config

  def from_config(config):
    return FACTMx_head_GMM_masked(**config)



class FACTMx_head_FlexTopicModel_masked(FACTMx_head_FlexTopicModel, FACTMx_head):
  head_type = 'FlexTopicModel_masked'

  def __init__(self,
               layer_configs={'preencoder':'linear'},
               masked_token=None,
               **super_kwargs):
    super().__init__(**super_kwargs)
    self.head_type = 'FlexTopicModel_masked'

    preencoder_config = layer_configs.pop('preencoder', 'linear')
    if preencoder_config == 'linear':
      self.layers['preencoder'] = tf.keras.Sequential(
                                        [tf.keras.Input(shape=(self.dim,)),
                                         tf.keras.layers.Dense(units=self.dim_latent,
                                                               kernel_initializer='orthogonal')]
                                      )
    else:
      self.layers['preencoder'] = tf.keras.Sequential.from_config(preencoder_config)

    if masked_token is None:
      masked_token = tf.random.normal(shape=(self.dim_latent,), stddev=.1)
    self.masked_token = tf.keras.Variable(masked_token)

    # get training variables
    self.t_vars = [*self.t_vars,
                   *self.layers['preencoder'].trainable_variables,
                   self.masked_token]


  def decode(self, latent, data):
    observed, mask = data

    return super().decode(latent, observed)


  def loss(self,
           data,
           latent,
           encoder_assignment_sample,
           encoder_assignment_logits,
           beta=1):
    observed, mask = data

    super_loss = super().loss(observed,
                              latent,
                              encoder_assignment_sample,
                              encoder_assignment_logits,
                              beta)

    return tf.reduce_sum([super_loss, *self.layers['preencoder'].losses])


  def encode(self, data):
    observed, mask = data

    encoder_dict = super().encode(observed)

    preencoded_input = encoder_dict['encoder_input']
    preencoded_input = self.layers['preencoder'](preencoded_input)
    masked_preencoded_input = (1-mask) * preencoded_input +\
                                mask * tf.expand_dims(self.masked_token, 0)

    encoder_dict['encoder_input'] = masked_preencoded_input

    return encoder_dict

  def get_config(self):
    config = super().get_config()
    config.update({
                'layer_configs': {key: layer.get_config() for key, layer in self.layers.items()},
                'masked_token': self.masked_token.numpy().tolist(),
             })
    return config

  def from_config(config):
    return FACTMx_head_FlexTopicModel_masked(**config)




class FACTMx_head_MultiNormal_masked(FACTMx_head_MultiNormal, FACTMx_head):
  head_type = 'MultiNormal_masked'

  def __init__(self,
               layer_configs={'preencoder':'linear'},
               masked_token=None,
               **super_kwargs):
    super().__init__(**super_kwargs)
    self.head_type = 'MultiNormal_masked'

    preencoder_config = layer_configs.pop('preencoder', 'linear')
    if preencoder_config == 'linear':
      self.layers['preencoder'] = tf.keras.Sequential(
                                        [tf.keras.Input(shape=(self.dim,)),
                                         tf.keras.layers.Dense(units=self.dim_latent,
                                                               kernel_initializer='orthogonal')]
                                      )
    else:
      self.layers['preencoder'] = tf.keras.Sequential.from_config(preencoder_config)

    if masked_token is None:
      masked_token = tf.random.normal(shape=(self.dim_latent,), stddev=.1)
    self.masked_token = tf.keras.Variable(masked_token)

    # get training variables
    self.t_vars = [*self.t_vars,
                   *self.layers['preencoder'].trainable_variables,
                   self.masked_token]


  def decode(self, latent, data):
    observed, mask = data

    return super().decode(latent, observed)


  def loss(self,
           data,
           latent,
           beta=1):
    observed, mask = data

    super_loss = super().loss(observed,
                              latent,
                              beta)

    return tf.reduce_sum([super_loss, *self.layers['preencoder'].losses])


  def encode(self, data):
    observed, mask = data

    encoder_dict = super().encode(observed)

    preencoded_input = encoder_dict['encoder_input']
    preencoded_input = self.layers['preencoder'](preencoded_input)
    masked_preencoded_input = (1-mask) * preencoded_input +\
                                mask * tf.expand_dims(self.masked_token, 0)

    encoder_dict['encoder_input'] = masked_preencoded_input

    return encoder_dict


  def get_config(self):
    config = super().get_config()
    config.update({
                'layer_configs': {key: layer.get_config() for key, layer in self.layers.items()},
                'masked_token': self.masked_token.numpy().tolist(),
             })
    return config

  def from_config(config):
    return FACTMx_head_MultiNormal_masked(**config)


# >>> Masked by token heads
class FACTMx_head_GMM_maskedZeros(FACTMx_head_GMM, FACTMx_head):
  head_type = 'GMM_maskedZeros'

  def __init__(self,
               masked_token=None,
               trainable=False,
               **super_kwargs):
    super().__init__(**super_kwargs)
    self.head_type = 'GMM_maskedZeros'
    self.trainable = trainable
                 
    if masked_token is None:
      masked_token = tf.zeros(shape=(self.dim,))
    if masked_token == 'random':
      masked_token = tf.keras.initializers.RandomNormal()(shape=(self.dim,))

    if trainable:
      self.masked_token = tf.keras.Variable(shape=(self.dim,))
      self.t_vars = tuple([*self.t_vars, self.masked_token])
    else:
      self.masked_token = tf.constant(masked_token)

  def decode(self, latent, data):
    observed, mask = data

    return super().decode(latent, observed)

  def loss(self,
           data,
           latent,
           encoder_assignment_sample,
           encoder_assignment_logits,
           beta=1):
    observed, mask = data

    return super().loss(observed,
                        latent,
                        encoder_assignment_sample,
                        encoder_assignment_logits,
                        beta)

  def encode(self, data):
    observed, mask = data

    encoder_dict = super().encode(observed)

    preencoded_input = encoder_dict['encoder_input']
    masked_preencoded_input = (1-mask) * preencoded_input +\
                                mask * tf.expand_dims(self.masked_token, 0)

    encoder_dict['encoder_input'] = masked_preencoded_input

    return encoder_dict

  def get_config(self):
    config = super().get_config()
    config.update({
                'masked_token': self.masked_token.numpy().tolist(),
                'trainable': self.trainable
             })
    return config

  def from_config(config):
    return FACTMx_head_GMM_maskedZeros(**config)


class FACTMx_head_FlexTopicModel_maskedZeros(FACTMx_head_FlexTopicModel, FACTMx_head):
  head_type = 'FlexTopicModel_maskedZeros'

  def __init__(self,
               masked_token=None,
               trainable=False,
               **super_kwargs):
    super().__init__(**super_kwargs)
    self.head_type = 'FlexTopicModel_maskedZeros'
    self.trainable = trainable
                 
    if masked_token is None:
      masked_token = tf.zeros(shape=(self.dim,))
    if masked_token == 'random':
      masked_token = tf.keras.initializers.RandomNormal()(shape=(self.dim,))

    if trainable:
      self.masked_token = tf.keras.Variable(shape=(self.dim,))
      self.t_vars = tuple([*self.t_vars, self.masked_token])
    else:
      self.masked_token = tf.constant(masked_token)

  def decode(self, latent, data):
    observed, mask = data

    return super().decode(latent, observed)

  def loss(self,
           data,
           latent,
           encoder_assignment_sample,
           encoder_assignment_logits,
           beta=1):
    observed, mask = data

    return super().loss(observed,
                        latent,
                        encoder_assignment_sample,
                        encoder_assignment_logits,
                        beta)

  def encode(self, data):
    observed, mask = data

    encoder_dict = super().encode(observed)

    preencoded_input = encoder_dict['encoder_input']
    masked_preencoded_input = (1-mask) * preencoded_input +\
                                mask * tf.expand_dims(self.masked_token, 0)

    encoder_dict['encoder_input'] = masked_preencoded_input

    return encoder_dict

  def get_config(self):
    config = super().get_config()
    config.update({
                'masked_token': self.masked_token.numpy().tolist(),
                'trainable': self.trainable
             })
    return config

  def from_config(config):
    return FACTMx_head_FlexTopicModel_maskedZeros(**config)


class FACTMx_head_MultiNormal_maskedZeros(FACTMx_head_MultiNormal, FACTMx_head):
  head_type = 'MultiNormal_maskedZeros'

  def __init__(self,
               masked_token=None,
               trainable=False,
               **super_kwargs):
    super().__init__(**super_kwargs)
    self.head_type = 'MultiNormal_maskedZeros'
    self.trainable = trainable
                 
    if masked_token is None:
      masked_token = tf.zeros(shape=(self.dim,))
    if masked_token == 'random':
      masked_token = tf.keras.initializers.RandomNormal()(shape=(self.dim,))

    if trainable:
      self.masked_token = tf.keras.Variable(shape=(self.dim,))
      self.t_vars = tuple([*self.t_vars, self.masked_token])
    else:
      self.masked_token = tf.constant(masked_token)
    

  def decode(self, latent, data):
    observed, mask = data

    return super().decode(latent, observed)

  def loss(self,
           data,
           latent,
           beta=1):
    observed, mask = data

    return super().loss(observed, latent, beta)

  def encode(self, data):
    observed, mask = data

    encoder_dict = super().encode(observed)

    preencoded_input = encoder_dict['encoder_input']
    masked_preencoded_input = (1-mask) * preencoded_input +\
                                mask * tf.expand_dims(self.masked_token, 0)

    encoder_dict['encoder_input'] = masked_preencoded_input

    return encoder_dict

  def get_config(self):
    config = super().get_config()
    config.update({
                'masked_token': self.masked_token.numpy().tolist(),
                'trainable': self.trainable
             })
    return config

  def from_config(config):
    return FACTMx_head_MultiNormal_maskedZeros(**config)
