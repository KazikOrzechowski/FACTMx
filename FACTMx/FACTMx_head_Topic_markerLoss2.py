import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Tuple

from FACTMx.FACTMx_head import FACTMx_head


class FACTMx_head_TopicModel_markerLoss(FACTMx_head):
  head_type='TopicModel_markerLoss'

  def __init__(self,
               dim, dim_latent, dim_words,
               head_name,
               layer_configs={'mixture_logits':'linear', 'encoder_classifier':'linear'},
               topic_profiles=None,
               marker_groups=None,
               eps=1E-10,
               temperature=1E-4,
               prop_loss_scale=1.,
               marker_loss_scale=1.,
               entropy_loss_scale=1.,):
    super().__init__(dim, dim_latent, head_name)
    self.dim_words = dim_words
    self.eps = eps
    self.temperature = temperature

    self.marker_groups = marker_groups
    self.prop_loss_scale = prop_loss_scale
    self.marker_loss_scale = marker_loss_scale
    self.entropy_loss_scale = entropy_loss_scale
    
    # >>> initialise layers >>>
    mixture_logits_config = layer_configs.pop('mixture_logits', 'linear')
    if mixture_logits_config == 'linear':
      self.layers['mixture_logits'] = tf.keras.Sequential(
                                        [tf.keras.Input(shape=(self.dim_latent,)),
                                         tf.keras.layers.Dense(units=self.dim,
                                                               activation='log_softmax',
                                                               kernel_initializer='orthogonal')]
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
                                                                   kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
                                                                   activation='log_softmax')]
                                          )
    else:
      self.layers['encoder_classifier'] = tf.keras.Sequential.from_config(encoder_classifier_config)

    assert self.layers['encoder_classifier'].input_shape == (None, None, self.dim_words)
    assert self.layers['encoder_classifier'].output_shape == (None, None, self.dim)
    # <<< initialise layers <<<

    #log proportions in topic profiles, with respect to fixed proportion of word0
    if topic_profiles is None:
      topic_profiles = tf.keras.initializers.Zeros()(shape=(dim_words, dim))
    self.topic_profiles_trainable = tf.keras.Variable(topic_profiles, 
                                                      trainable=True,
                                                      dtype=tf.float32,
                                                      name=f'{head_name}_topics')

    self.t_vars = [*self.layers['mixture_logits'].trainable_variables,
                   *self.layers['encoder_classifier'].trainable_variables,
                   self.topic_profiles_trainable]


  def get_assignment_distribution(self, logits):
    return tfp.distributions.RelaxedOneHotCategorical(logits=logits,
                                                      temperature=self.temperature)

  
  def get_log_topic_profiles(self):
    return tf.math.log_softmax(self.topic_profiles_trainable, axis=0)

  def get_topic_profiles(self):
    return tf.math.exp(self.get_log_topic_profiles())

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


  def loss(self, 
           data, 
           latent, 
           encoder_assignment_sample, 
           encoder_assignment_logits, 
           beta=1):
    batch_size = data.shape[0]
             
    _, assignment_logits, log_topic_proportions = FACTMx_head_TopicModel_markerLoss.decode(self, latent, data, sample=False)

    q_logits = tf.math.subtract(assignment_logits, log_topic_proportions)

    kl_divergence = tf.reduce_sum(
        tfp.distributions.OneHotCategorical(logits=encoder_assignment_logits).kl_divergence(
            tfp.distributions.OneHotCategorical(logits=log_topic_proportions)
            )
    )
    kl_loss = self.prop_loss_scale * kl_divergence / batch_size

    #if np.random.choice([True, False]):
    if True:
      probs = encoder_assignment_sample
    else:
      probs = tf.math.softmax(encoder_assignment_logits, axis=-1)
      
    log_likelihood = tf.reduce_sum(
        tf.math.multiply(probs, q_logits),
        #axis=2
    )
    ll_loss = -log_likelihood / batch_size

    marker_loss = tf.constant(0.)
    if self.marker_groups is not None:
      counts_data = tf.expand_dims(data, axis=-2)
      markers = [tf.reduce_mean(tf.gather(counts_data, marker_inds, axis=-1), axis=-1) for marker_inds, _ in self.marker_groups]
      antagonists = [tf.reduce_mean(tf.gather(counts_data, antagonist_inds, axis=-1), axis=-1) for _, antagonist_inds in self.marker_groups]

      marker_loss = tf.reduce_mean(tf.stack(markers) * tf.stack(antagonists) * tf.expand_dims(probs, axis=0))

    entropy_loss = tf.constant(0.)
    if self.entropy_loss_scale != 0.:
      entropy = tf.math.softmax(encoder_assignment_logits, axis=-1) + self.eps
      entropy = tf.reduce_mean(entropy, axis=(0,1))
      entropy = entropy * tf.math.log(entropy)
      entropy_loss = tf.reduce_sum(entropy) * self.entropy_loss_scale #entropy loss is -entropy, since we want a mixed assignment
    
    return tf.reduce_sum([kl_loss, 
                          ll_loss,
                          marker_loss,
                          entropy_loss,
                          *self.layers['mixture_logits'].losses,
                          *self.layers['encoder_classifier'].losses])


  def encode(self, data):
    assignment_logits = self.layers['encoder_classifier'](data) 
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
        'marker_groups':self.marker_groups,
        'temperature':self.temperature,
        'eps':self.eps,
        'topic_profiles':self.topic_profiles_trainable.numpy().tolist(),
        "layer_configs": {key: layer.get_config() for key, layer in self.layers.items()},
        'prop_loss_scale':self.prop_loss_scale,
        'marker_loss_scale':self.marker_loss_scale,
        'entropy_loss_scale':self.entropy_loss_scale,
    }
    return config

  def from_config(config):
    return FACTMx_head_TopicModel_markerLoss(**config)
