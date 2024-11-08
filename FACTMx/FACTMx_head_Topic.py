import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from typing import Tuple

from . import FACTMx_head

class FACTMx_head_TopicModel(FACTMx_head):
  head_type='TopicModel'

  def __init__(self,
               dim, dim_latent, dim_words,
               head_name,
               decode_config='linear',
               topic_profiles=None,
               temperature=1E-4):
    super().__init__(dim, dim_latent, head_name)
    self.eps = 1E-5
    self.dim_words = dim_words
    self.temperature = temperature

    if decode_config == 'linear':
      self.decode_model = tf.keras.Sequential(
                            [tf.keras.Input(shape=(self.dim_latent,)),
                             tf.keras.layers.Dense(units=self.dim,
                                                   kernel_initializer='orthogonal')]
                          )
    else:
      self.decode_model = tf.keras.Sequential.from_config(decode_config)

    assert self.decode_model.output_shape == (None, self.dim)
    assert self.decode_model.input_shape == (None, self.dim_latent)

    #log proportions in topic profiles, with respect to fixed proportion of word0
    if topic_profiles is None:
      topic_profiles = tf.keras.initializers.Orthogonal()(shape=(dim_words-1, dim+1))
    self.topic_profiles_trainable = tf.keras.Variable(topic_profiles, trainable=True)

    self.t_vars = [*self.decode_model.trainable_variables,
                   self.topic_profiles_trainable]


  def get_assignment_distribution(self, logits):
    return tfp.distributions.RelaxedOneHotCategorical(logits=logits,
                                                      temperature=self.temperature)


  def get_log_topic_profiles(self):
    paddings_profiles = tf.constant([[1, 0], [0, 0]])
    log_topic_profiles = tf.pad(self.topic_profiles_trainable,
                                paddings_profiles,
                                'CONSTANT')
    log_topic_profiles = tf.keras.activations.log_softmax(log_topic_profiles,
                                                          axis=0)
    return log_topic_profiles


  def decode_log_topic_proportions(self, latent):
    paddings_proportions = tf.constant([[0, 0], [1, 0]])
    log_topic_proportions = tf.pad(self.decode_model(latent),
                                   paddings_proportions,
                                   'CONSTANT')
    log_topic_proportions = tf.keras.activations.log_softmax(log_topic_proportions,
                                                             axis=-1)
    return log_topic_proportions


  def decode(self, latent, data):
    log_topic_proportions = self.decode_log_topic_proportions(latent)
    log_topic_proportions = tf.reshape(log_topic_proportions,
                                       (-1, 1, self.dim+1))

    log_topic_profiles = self.get_log_topic_profiles()

    assignment_logits = tf.math.add(log_topic_proportions, tf.matmul(data, log_topic_profiles))
    assignment_sample = self.get_assignment_distribution(assignment_logits).sample()

    return assignment_sample, assignment_logits, log_topic_proportions.numpy()


  def loss(self, data, latent, assignment_sample, beta=1):
    _, assignment_logits, log_topic_props = self.decode(latent, data)

    q_logits = tf.math.subtract(assignment_logits, log_topic_props)

    kl_divergence = tf.reduce_mean(
        tfp.distributions.OneHotCategorical(logits=q_logits).kl_divergence(
            tfp.distributions.OneHotCategorical(logits=log_topic_props)
            ),
        axis=-1
    )

    log_likelihood = tf.reduce_sum(
        tf.math.multiply(assignment_sample, q_logits),
        axis=2
    )

    log_likelihood = tf.reduce_mean(log_likelihood)

    return tf.reduce_sum([beta*kl_divergence, 
                          -log_likelihood,
                          *self.decode_model.losses])


  def encode(self, data):
    log_topic_profiles = self.get_log_topic_profiles()

    assignment_logits = tf.matmul(data, log_topic_profiles)
    assignment_sample = self.get_assignment_distribution(assignment_logits).sample()

    proportions_sample = tf.reduce_mean(assignment_sample, axis=1) + self.eps
    log_proportions_sample = tf.math.log(proportions_sample)

    encoder_input = log_proportions_sample[:,1:] - tf.reshape(log_proportions_sample[:,0], (-1, 1))

    return {'encoder_input': encoder_input,
            'assignment_sample': assignment_sample}


  def get_config(self):
    config = {
        'dim':self.dim,
        'dim_latent':self.dim_latent,
        'dim_words':self.dim_words,
        'head_name':self.head_name,
        'head_type':self.head_type,
        'temperature':self.temperature,
        'topic_profiles':self.topic_profiles_trainable.numpy().tolist(),
        'decode_config':self.decode_model.get_config()
    }
    return config

  def from_config(config):
    return FACTMx_head_TopicModel(**config)

  def save_weights(self, head_path):
    self.decode_model.save_weights(f'{head_path}.weights.h5')

  def load_weights(self, head_path):
    self.decode_model.load_weights(f'{head_path}.weights.h5')
