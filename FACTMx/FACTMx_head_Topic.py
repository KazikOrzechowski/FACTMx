"""
Topic-model data head for FACTMx.

The topic head treats each sub-observation as a bag/vector of word counts and
learns topic profiles together with latent-dependent topic proportions.  The
encoder classifier provides relaxed topic assignments so the ELBO can be trained
with gradient-based optimisation.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Tuple

from FACTMx.FACTMx_head import FACTMx_head


class FACTMx_head_TopicModel(FACTMx_head):
  """
  Topic-model head for document-like count vectors.
  
  The decoder maps latent points to topic proportions.  Learned topic profiles
  map topic assignments to word likelihoods for each observation.
  """
  head_type='TopicModel'

  def __init__(self,
               dim, dim_latent, dim_words,
               head_name,
               layer_configs={'mixture_logits':'linear', 'encoder_classifier':'linear'},
               topic_profiles=None,
               eps=1E-3,
               temperature=1E-4):
    """Initialise topic proportion layers, encoder classifier, and topic profiles."""
    super().__init__(dim, dim_latent, head_name)
                 
    self.eps = eps
    self.dim_words = dim_words
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
                                            [tf.keras.Input(shape=(None, self.dim_words)),
                                             tf.keras.layers.Dense(units=self.dim,
                                                                   activation='log_softmax',
                                                                   bias_initializer='ones')]
                                          )
    else:
      self.layers['encoder_classifier'] = tf.keras.Sequential.from_config(encoder_classifier_config)

    assert self.layers['encoder_classifier'].input_shape == (None, None, self.dim_words)
    assert self.layers['encoder_classifier'].output_shape == (None, None, self.dim)
    # <<< initialise layers <<<

    # Topic profiles are stored in log space so matrix products in decode() can
    # accumulate word log-likelihoods directly.
    if topic_profiles is None:
      topic_profiles = tf.keras.initializers.RandomNormal()(shape=(dim_words, dim))
    self.topic_profiles_trainable = tf.keras.Variable(topic_profiles, 
                                                      trainable=True,
                                                      dtype=tf.float32)

    self.t_vars = [*self.layers['mixture_logits'].trainable_variables,
                   *self.layers['encoder_classifier'].trainable_variables,
                   self.topic_profiles_trainable]


  def get_assignment_distribution(self, logits):
    """Return a relaxed categorical distribution over topic assignments."""
    return tfp.distributions.RelaxedOneHotCategorical(logits=logits,
                                                      temperature=self.temperature)

  def get_log_topic_profiles(self):
    """Return trainable log topic profiles."""
    return self.topic_profiles_trainable

  def get_topic_profiles(self):
    """Return exponentiated topic profiles in probability space."""
    return tf.math.exp(self.get_log_topic_profiles())

  def decode_log_topic_proportions(self, latent):
    """Predict stabilized log topic proportions from latent points."""
    log_topic_proportions = self.layers['mixture_logits'](latent) 

    #minimal topic proportions should be around eps
    log_eps = tf.constant(tf.math.log(self.eps), shape=log_topic_proportions.shape)
    return tf.reduce_logsumexp(tf.stack([log_topic_proportions, log_eps]), axis=0)

  def decode(self, latent, data, sample=True):
    """Return topic assignment samples/logits and decoded topic proportions."""
    log_topic_proportions = self.decode_log_topic_proportions(latent)
    log_topic_proportions = tf.reshape(log_topic_proportions, (-1, 1, self.dim))

    log_topic_profiles = self.get_log_topic_profiles()

    log_likelihoods = tf.matmul(data, log_topic_profiles)

    # Topic assignment logits combine latent topic proportions with the observed
    # word likelihood under each trainable topic profile.
    assignment_logits = tf.math.add(log_topic_proportions, log_likelihoods)
    assignment_sample = self.get_assignment_distribution(assignment_logits).sample() if sample else None

    return assignment_sample, assignment_logits, log_topic_proportions

  def loss(self, 
           data, 
           latent, 
           encoder_assignment_sample, 
           encoder_assignment_logits, 
           beta=1):
    """Compute assignment KL plus expected negative topic log-likelihood."""
    _, assignment_logits, log_topic_proportions = FACTMx_head_TopicModel.decode(self, latent, data, sample=False)

    q_logits = tf.math.subtract(assignment_logits, log_topic_proportions)

    kl_divergence = tf.reduce_mean(
        tfp.distributions.OneHotCategorical(logits=encoder_assignment_logits).kl_divergence(
            tfp.distributions.OneHotCategorical(logits=log_topic_proportions)
            )
    )

    log_likelihood = tf.reduce_sum(
        tf.math.multiply(encoder_assignment_sample, q_logits),
    )
    batch_size, subbatch_size, _ = data.shape

    ll_loss = -log_likelihood / batch_size / subbatch_size
    
    return tf.reduce_sum([kl_divergence,
                          ll_loss,
                          *self.layers['mixture_logits'].losses,
                          *self.layers['encoder_classifier'].losses])


  def encode(self, data):
    """Infer relaxed topic assignments and produce encoder log proportions."""
    assignment_logits = self.layers['encoder_classifier'](data) 
    assignment_sample = self.get_assignment_distribution(assignment_logits).sample() 

    proportions_sample = tf.reduce_mean(assignment_sample, axis=1) + self.eps
    encoder_input = tf.math.log(proportions_sample)

    return {'encoder_input': encoder_input,
            'encoder_assignment_sample': assignment_sample,
            'encoder_assignment_logits': assignment_logits}


  def get_config(self):
    """Serialize topic head dimensions, layer configs, and topic profiles."""
    config = {
        'dim':self.dim,
        'dim_latent':self.dim_latent,
        'dim_words':self.dim_words,
        'head_name':self.head_name,
        'head_type':self.head_type,
        'temperature':self.temperature,
        'eps':self.eps,
        'topic_profiles':self.topic_profiles_trainable.numpy().tolist(),
        "layer_configs": {key: layer.get_config() for key, layer in self.layers.items()},
    }
    return config

  def from_config(config):
    """Reconstruct a topic head from a config dictionary."""
    return FACTMx_head_TopicModel(**config)
