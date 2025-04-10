import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Tuple

try:
  import tensorflow_model_optimization as tfmot
  from tensorflow_model_optimization.python.core.keras.compat import keras
  _TFMOT_IS_LOADED = True
except ImportError:
  import tensorflow.keras as keras
  _TFMOT_IS_LOADED = False

from FACTMx.FACTMx_head import FACTMx_head


def ragged_mat_mul(ragged_tensor, matrix):
  output_signature = tf.RaggedTensorSpec(shape=[None, None],
                                         ragged_rank=0)
  ragged_mul_function = lambda x: tf.cast(tf.reshape(x, (-1, matrix.shape[0])), tf.float32) @ tf.cast(matrix, tf.float32)
  
  return tf.map_fn(
    ragged_mul_function,
    ragged_tensor,
    fn_output_signature=output_signature
  )

def ragged_classifier_pass(ragged_tensor, model):
  output_signature = tf.RaggedTensorSpec(shape=[None, None],
                                         ragged_rank=0)

  model_shape = (1, -1, model.input_shape[-1])
  ragged_pass = lambda tensor: tf.reshape(
    model(tf.reshape(tensor, model_shape)),
    (-1, model.output_shape[-1])
  )
  
  return tf.map_fn(
    ragged_pass,
    ragged_tensor,
    fn_output_signature=output_signature
  )

def ragged_KL_divergence(ragged_logits, 
                         second_logits, 
                         distribution_function):
  output_signature = tf.RaggedTensorSpec(shape=[None, ], 
                                         dtype=tf.float32,
                                         ragged_rank=0)

  ragged_KL = lambda x: distribution_function(logits=x[0]).kl_divergence(distribution_function(logits=x[1]))

  return tf.map_fn(
    ragged_KL,
    tf.stack([ragged_logits, second_logits], axis=1),
    fn_output_signature=output_signature
  )



class FACTMx_head_FlexTopicModel(FACTMx_head):
  head_type='FlexTopicModel'

  def __init__(self,
               dim, dim_latent, dim_words,
               head_name,
               decode_config='linear',
               encoder_classifier_config='linear',
               ragged=False,
               topic_profiles=None,
               topic_L2_penalty=None,
               proportions_L2_penalty=None,
               prop_loss_scale=1.,
               pruning_params={'encoder':None, 'decoder':None},
               eps=1E-3,
               temperature=1E-4):
    super().__init__(dim, dim_latent, head_name)
    self.eps = eps
    self.dim_words = dim_words
    self.ragged = ragged
    self.topic_L2_penalty = topic_L2_penalty
    self.proportions_L2_penalty = proportions_L2_penalty
    self.prop_loss_scale = prop_loss_scale
    self.pruning_params = pruning_params
    self.temperature = temperature

    if decode_config == 'linear':
      self.decode_model = tf.keras.Sequential(
                            [tf.keras.Input(shape=(self.dim_latent,)),
                             tf.keras.layers.Dense(units=self.dim,
                                                   kernel_initializer='orthogonal')]
                          )
    else:
      self.decode_model = tf.keras.Sequential.from_config(decode_config)

    assert self.decode_model.input_shape == (None, self.dim_latent)
    assert self.decode_model.output_shape == (None, self.dim)

    decoder_pruning = pruning_params.pop('decoder', None)
    if decoder_pruning is not None and _TFMOT_IS_LOADED:
      self.decode_model = tfmot.sparsity.keras.prune_low_magnitude(self.decode_model,
                                                                   pruning_schedule=tfmot.sparsity.keras.PruningSchedule.from_config(decoder_pruning))

    if encoder_classifier_config == 'linear':
      self.encoder_classifier = tf.keras.Sequential(
                                  [tf.keras.Input(shape=(None, self.dim_words)), 
                                   tf.keras.layers.Dense(units=self.dim+1,
                                                         activation='log_softmax')]
                                )
    else:
      self.encoder_classifier = tf.keras.Sequential.from_config(encoder_classifier_config)

    encoder_pruning = pruning_params.pop('encoder', None)
    if encoder_pruning is not None and _TFMOT_IS_LOADED:
      self.encoder_classifier = tfmot.sparsity.keras.prune_low_magnitude(self.encoder_classifier,
                                                                         pruning_schedule=tfmot.sparsity.keras.PruningSchedule.from_config(encoder_pruning))

    assert self.encoder_classifier.input_shape == (None, None, self.dim_words)
    assert self.encoder_classifier.output_shape == (None, None, self.dim+1)

    #log proportions in topic profiles, with respect to fixed proportion of word0
    if topic_profiles is None:
      topic_profiles = tf.keras.initializers.Orthogonal()(shape=(dim_words-1, dim+1))
    self.topic_profiles_trainable = tf.Variable(topic_profiles, 
                                                trainable=True,
                                                dtype=tf.float32)

    self.t_vars = [*self.decode_model.trainable_variables,
                   *self.encoder_classifier.trainable_variables,
                   self.topic_profiles_trainable]


  def get_assignment_distribution(self, logits):
    return tfp.distributions.RelaxedOneHotCategorical(logits=logits,
                                                      temperature=self.temperature)

  
  def get_ragged_assignments(self, ragged_logits):
    output_signature = tf.RaggedTensorSpec(shape=[None, None], 
                                           dtype=tf.float32,
                                           ragged_rank=0)

    ragged_sampling = lambda x: self.get_assignment_distribution(x).sample()

    return tf.map_fn(
      ragged_sampling,
      ragged_logits,
      fn_output_signature=output_signature
    )

  
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
    paddings_proportions = tf.constant([[0, 0], [1, 0]])
    log_topic_proportions = tf.pad(self.decode_model(latent),
                                   paddings_proportions,
                                   'CONSTANT')
    log_topic_proportions = tf.math.log(
      tf.keras.activations.softmax(log_topic_proportions, axis=-1)
    )

    #minimal topic proportions should be around eps
    log_eps = tf.constant(tf.math.log(self.eps), shape=log_topic_proportions.shape)
    return tf.reduce_logsumexp(tf.stack([log_topic_proportions, log_eps]), axis=0)


  def decode(self, latent, data):
    log_topic_proportions = self.decode_log_topic_proportions(latent)
    log_topic_proportions = tf.reshape(log_topic_proportions,
                                       (-1, 1, self.dim+1))

    log_topic_profiles = self.get_log_topic_profiles()

    log_likelihoods = tf.matmul(data, log_topic_profiles) if not self.ragged else ragged_mat_mul(data, log_topic_profiles)

    assignment_logits = tf.math.add(log_topic_proportions, log_likelihoods)
    assignment_sample = self.get_assignment_distribution(assignment_logits).sample() if not self.ragged else self.get_ragged_assignments(assignment_logits)

    return assignment_sample, assignment_logits, log_topic_proportions


  def loss(self, 
           data, 
           latent, 
           encoder_assignment_sample, 
           encoder_assignment_logits, 
           beta=1):
    _, assignment_logits, log_topic_proportions = self.decode(latent, data)

    q_logits = tf.math.subtract(assignment_logits, log_topic_proportions)

    if not self.ragged:
      kl_divergence = tf.reduce_sum(
          tfp.distributions.OneHotCategorical(logits=encoder_assignment_logits).kl_divergence(
              tfp.distributions.OneHotCategorical(logits=log_topic_proportions)
              )
      )
    else:
      kl_divergence = tf.reduce_sum(
        ragged_KL_divergence(encoder_assignment_logits, 
                             log_topic_proportions, 
                             tfp.distributions.OneHotCategorical)
      )

    log_likelihood = tf.reduce_sum(
        tf.math.multiply(encoder_assignment_sample, q_logits),
        #axis=2
    )
    #log_likelihood = tf.reduce_mean(log_likelihood)
    batch_size = data.shape[0]
    
    return tf.reduce_sum([self.prop_loss_scale*kl_divergence/batch_size, 
                          -log_likelihood/batch_size,
                          self.get_topic_regularization_loss(),
                          self.get_proportions_regularization_loss(log_topic_proportions),
                          *self.decode_model.losses])


  def encode(self, data):
    assignment_logits = self.encoder_classifier(data) if not self.ragged else ragged_classifier_pass(data, self.encoder_classifier)
    assignment_sample = self.get_assignment_distribution(assignment_logits).sample() if not self.ragged else self.get_ragged_assignments(assignment_logits)

    proportions_sample = tf.reduce_mean(assignment_sample, axis=1) + self.eps
    log_proportions_sample = tf.math.log(proportions_sample)

    encoder_input = log_proportions_sample[:,1:] - tf.reshape(log_proportions_sample[:,0], (-1, 1))

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
        'ragged':self.ragged,
        'temperature':self.temperature,
        'eps':self.eps,
        'prop_loss_scale':self.prop_loss_scale,
        'pruning_params':self.pruning_params,
        'topic_profiles':self.topic_profiles_trainable.numpy().tolist(),
        'topic_L2_penalty':self.topic_L2_penalty,
        'proportions_L2_penalty':self.proportions_L2_penalty,
        'encoder_classifier_config':self.encoder_classifier.get_config(),
        'decode_config':self.decode_model.get_config()
    }
    return config

  def from_config(config):
    return FACTMx_head_TopicModel(**config)

  def save_weights(self, head_path):
    self.decode_model.save_weights(f'{head_path}_decode_model.weights.h5')
    self.encoder_classifier.save_weights(f'{head_path}_encoder_classifier.weights.h5')

  def load_weights(self, head_path):
    self.decode_model.load_weights(f'{head_path}_decode_model.weights.h5')
    self.encoder_classifier.load_weights(f'{head_path}_encoder_classifier.weights.h5')
