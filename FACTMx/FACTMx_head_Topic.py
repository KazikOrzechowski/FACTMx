import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Tuple

from FACTMx.FACTMx_head import FACTMx_head


class FACTMx_head_TopicModel(FACTMx_head):
  head_type='TopicModel'

  def __init__(self,
               dim, dim_latent, dim_words, dim_first_pass,
               head_name,
               layer_configs={'mixture_logits':'linear', 'encoder_classifier':'linear'},
               topic_profiles=None,
               topic_L2_penalty=None,
               proportions_L2_penalty=None,
               prop_loss_scale=1.,
               eps=1E-3,
               temperature=1E-4):
    super().__init__(dim, dim_latent, head_name)
                 
    self.eps = eps
    self.dim_words = dim_words
    self.dim_first_pass = dim_first_pass
    self.topic_L2_penalty = topic_L2_penalty
    self.proportions_L2_penalty = proportions_L2_penalty
    self.prop_loss_scale = prop_loss_scale
    self.temperature = temperature
    
    # >>> initialise layers >>>
    mixture_logits_config = layer_configs.pop('mixture_logits', 'linear')
    if mixture_logits_config == 'linear':
      self.layers['mixture_logits'] = tf.keras.Sequential(
                                        [tf.keras.Input(shape=(self.dim_latent,)),
                                         tf.keras.layers.Dense(units=self.dim,
                                                               activation='log_softmax',
                                                               kernel_initializer='orthogonal',
                                                               bias_initializer='ones')]
                                      )
    else:
      self.layers['mixture_logits'] = tf.keras.Sequential.from_config(mixture_logits_config)

    assert self.layers['mixture_logits'].output_shape == (None, self.dim)
    assert self.layers['mixture_logits'].input_shape == (None, self.dim_latent)

                 
    encoder_classifier_config = layer_configs.pop('encoder_classifier', 'linear')
    if encoder_classifier_config == 'linear':
      self.layers['encoder_classifier'] = tf.keras.Sequential(
                                            [tf.keras.Input(shape=(None, self.dim_words + self.dim_first_pass)),
                                             tf.keras.layers.Dense(units=self.dim,
                                                                   activation='log_softmax',
                                                                   bias_initializer='ones')]
                                          )
    else:
      self.layers['encoder_classifier'] = tf.keras.Sequential.from_config(encoder_classifier_config)

    assert self.layers['encoder_classifier'].input_shape == (None, None, self.dim_words + self.dim_first_pass)
    assert self.layers['encoder_classifier'].output_shape == (None, None, self.dim)

    
    first_classifier_config = layer_configs.pop('first_classifier', 'linear')
    if first_classifier_config == 'linear':
      self.layers['first_classifier'] = tf.keras.Sequential(
                                            [tf.keras.Input(shape=(None, self.dim_normal)),
                                             tf.keras.layers.Dense(units=self.dim,
                                                                   activation='log_softmax')]
                                          )
    else:
      self.layers['first_classifier'] = tf.keras.Sequential.from_config(first_classifier_config)

    assert self.layers['first_classifier'].input_shape == (None, None, self.dim_words)
    assert self.layers['first_classifier'].output_shape == (None, None, self.dim)
    # <<< initialise layers <<<

    #log proportions in topic profiles, with respect to fixed proportion of word0
    if topic_profiles is None:
      topic_profiles = tf.keras.initializers.RandomNormal()(shape=(dim_words-1, dim))
    self.topic_profiles_trainable = tf.keras.Variable(topic_profiles, 
                                                      trainable=True,
                                                      dtype=tf.float32)

    self.t_vars = [*self.layers['mixture_logits'].trainable_variables,
                   *self.layers['encoder_classifier'].trainable_variables,
                   *self.layers['first_classifier'].trainable_variables,
                   self.topic_profiles_trainable]


  def get_assignment_distribution(self, logits):
    return tfp.distributions.RelaxedOneHotCategorical(logits=logits,
                                                      temperature=self.temperature)

  
  def get_log_topic_profiles(self):
    paddings_profiles = tf.constant([[1, 0], [0, 0]])
    log_topic_profiles = tf.pad(self.topic_profiles_trainable,
                                paddings_profiles,
                                'CONSTANT')
    log_topic_profiles = tf.math.log(
      tf.keras.activations.softmax(log_topic_profiles, axis=0)
    )
    return log_topic_profiles

  def get_topic_profiles(self):
    return tf.math.exp(self.get_log_topic_profiles())

  def get_topic_regularization_loss(self):
    if self.topic_L2_penalty is None:
      return tf.constant(0.)
    else:
      return self.topic_L2_penalty * tf.reduce_sum(self.get_topic_profiles() ** 2)

  def get_proportions_regularization_loss(self, log_topic_proportions):
    if self.proportions_L2_penalty is None:
      return tf.constant(0.)
    else:
      return self.proportions_L2_penalty * tf.reduce_sum(tf.math.exp(2 * log_topic_proportions))

  def decode_log_topic_proportions(self, latent):
    log_topic_proportions = self.layers['mixture_logits'](latent) 

    #minimal topic proportions should be around eps
    log_eps = tf.constant(tf.math.log(self.eps), shape=log_topic_proportions.shape)
    return tf.reduce_logsumexp(tf.stack([log_topic_proportions, log_eps]), axis=0)

  def decode(self, latent, data, sample=True):
    log_topic_proportions = self.decode_log_topic_proportions(latent)
    log_topic_proportions = tf.reshape(log_topic_proportions, (-1, 1, self.dim))

    log_topic_profiles = self.get_log_topic_profiles()

    log_likelihoods = tf.matmul(data, log_topic_profiles)

    assignment_logits = tf.math.add(log_topic_proportions, log_likelihoods)
    assignment_sample = self.get_assignment_distribution(assignment_logits).sample() if sample else None

    return assignment_sample, assignment_logits, log_topic_proportions

  # last working
  # def loss(self, 
  #          data, 
  #          latent, 
  #          encoder_assignment_sample, 
  #          encoder_assignment_logits, 
  #          beta=1):
  #   _, assignment_logits, log_topic_proportions = FACTMx_head_TopicModel.decode(self, latent, data, sample=False)

  #   q_logits = tf.math.subtract(assignment_logits, log_topic_proportions)

  #   kl_divergence = tf.reduce_sum(
  #       tfp.distributions.OneHotCategorical(logits=encoder_assignment_logits).kl_divergence(
  #           tfp.distributions.OneHotCategorical(logits=log_topic_proportions)
  #           )
  #   )

  #   log_likelihood = tf.reduce_sum(
  #       tf.math.multiply(encoder_assignment_sample, q_logits),
  #       #axis=2
  #   )
  #   #log_likelihood = tf.reduce_mean(log_likelihood)
  #   batch_size = data.shape[0]
    
  #   return tf.reduce_sum([self.prop_loss_scale*kl_divergence/batch_size, 
  #                         -log_likelihood/batch_size,
  #                         self.get_topic_regularization_loss(),
  #                         self.get_proportions_regularization_loss(log_topic_proportions),
  #                         *self.layers['mixture_logits'].losses,
  #                         *self.layers['encoder_classifier'].losses])

  def loss(self, 
           data, 
           latent, 
           encoder_assignment_sample, 
           encoder_assignment_logits, 
           beta=1):
    _, assignment_logits, log_topic_proportions = FACTMx_head_TopicModel.decode(self, latent, data, sample=False)

    q_logits = tf.math.subtract(assignment_logits, log_topic_proportions)

    encoder_probs = tf.math.softmax(encoder_assignment_logits)
    mean_probs = tf.reduce_mean(encoder_probs, axis=1, keepdims=True) + 1E-50
    kl_divergence = tf.reduce_mean(
        tfp.distributions.OneHotCategorical(probs=mean_probs).kl_divergence(
            tfp.distributions.OneHotCategorical(logits=log_topic_proportions)
            )
    )

    log_likelihood = tf.reduce_sum(
        tf.math.multiply(encoder_probs, q_logits),
        #axis=2
    )
    #log_likelihood = tf.reduce_mean(log_likelihood)
    batch_size, subbatch_size, _ = data.shape
    
    return tf.reduce_sum([self.prop_loss_scale*kl_divergence, 
                          -log_likelihood / batch_size / subbatch_size,
                          self.get_topic_regularization_loss(),
                          self.get_proportions_regularization_loss(log_topic_proportions),
                          *self.layers['mixture_logits'].losses,
                          *self.layers['encoder_classifier'].losses])


  def encode(self, data, first_pass):
    _, subbatch_size, _ = data.shape
    first_pass = tf.expand_dims(first_pass, 1)
    first_pass = tf.repeat(first_pass, subbatch_size, axis=1)
    input = tf.concat([data, first_pass], axis=-1)
    
    assignment_logits = self.layers['encoder_classifier'](input) 
    assignment_sample = self.get_assignment_distribution(assignment_logits).sample() 

    proportions_sample = tf.reduce_mean(assignment_sample, axis=1) + self.eps
    encoder_input = tf.math.log(proportions_sample)

    return {'encoder_input': encoder_input,
            'encoder_assignment_sample': assignment_sample,
            'encoder_assignment_logits': assignment_logits}
  
  def first_pass(self, data):
    assignment_logits = self.layers['first_classifier'](data) 
    assignment_sample = self.get_assignment_distribution(assignment_logits).sample() 

    proportions_sample = tf.reduce_mean(assignment_sample, axis=1) + self.eps
    encoder_input = tf.math.log(proportions_sample)

    return {'encoder_input': encoder_input,
            'encoder_assignment_sample': assignment_sample,
            'encoder_assignment_logits': assignment_logits}

  def get_config(self):
    config = {
        'dim':self.dim,
        'dim_latent':self.dim_latent,
        'dim_words':self.dim_words,
        'head_name':self.head_name,
        'head_type':self.head_type,
        'temperature':self.temperature,
        'eps':self.eps,
        'prop_loss_scale':self.prop_loss_scale,
        'topic_profiles':self.topic_profiles_trainable.numpy().tolist(),
        'topic_L2_penalty':self.topic_L2_penalty,
        'proportions_L2_penalty':self.proportions_L2_penalty,
        "layer_configs": {key: layer.get_config() for key, layer in self.layers.items()},
    }
    return config

  def from_config(config):
    return FACTMx_head_TopicModel(**config)
