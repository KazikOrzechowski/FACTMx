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

from FACTMx.FACTMx_head import FACTMx_head


class FACTMx_head_ZINB_mixture2(FACTMx_head):
  head_type = 'ZINB_mixture2'

  def __init__(self,
               dim, dim_latent, dim_counts,
               head_name,
               layer_configs={'mixture_logits':'linear', 'encoder_classifier':'linear'},
               mixture_params={'logits':None, 'log_count_spread':None, 'inflated_loc_logits':None},
               temperature=1E-4,
               eps=1E-3,
               prop_loss_scale=1.):
    super().__init__(dim, dim_latent, head_name)

    self.dim_mixtures = dim + 1
    self.dim_counts = dim_counts
    self.temperature = temperature
    self.eps = eps
    self.prop_loss_scale = prop_loss_scale

    # >>> initialise layers >>>
    mixture_logits_config = layer_configs.pop('mixture_logits', 'linear')
    if mixture_logits_config == 'linear':
      self.layers['mixture_logits'] = tf.keras.Sequential(
                                        [tf.keras.Input(shape=(self.dim_latent,)),
                                         tf.keras.layers.Dense(units=self.dim,
                                                               kernel_initializer='orthogonal')]
                                      )
    else:
      self.layers['mixture_logits'] = tf.keras.Sequential.from_config(mixture_logits_config)

    assert self.layers['mixture_logits'].output_shape == (None, self.dim)
    assert self.layers['mixture_logits'].input_shape == (None, self.dim_latent)

    encoder_classifier_config = layer_configs.pop('encoder_classifier', 'linear')
    if encoder_classifier_config == 'linear':
      self.layers['encoder_classifier'] = tf.keras.Sequential(
                                            [tf.keras.Input(shape=(None, self.dim_counts)),
                                             tf.keras.layers.Dense(units=self.dim_mixtures,
                                                                   activation='log_softmax')]
                                          )
    else:
      self.layers['encoder_classifier'] = tf.keras.Sequential.from_config(encoder_classifier_config)

    assert self.layers['encoder_classifier'].input_shape == (None, None, self.dim_counts)
    assert self.layers['encoder_classifier'].output_shape == (None, None, self.dim_mixtures)
    # <<< initialise layers <<<

    # >>> initialise mixtures >>>
    logits = mixture_params.pop('logits', None)
    if logits is None:
      logits = tf.keras.initializers.RandomNormal()(shape=(self.dim_mixtures, self.dim_counts))

    self.logits = tf.keras.Variable(logits,
                                    trainable=True,
                                    dtype=tf.float32)

    log_count_spread = mixture_params.pop('log_count_spread', None)
    if log_count_spread is None:
      log_count_spread = tf.keras.initializers.Zeros()(shape=(self.dim_mixtures, self.dim_counts))

    self.log_count_spread = tf.keras.Variable(log_count_spread,
                                              trainable=True,
                                              dtype=tf.float32)

    inflated_loc_logits = mixture_params.pop('inflated_loc_logits', None)
    if inflated_loc_logits is None:
      inflated_loc_logits = tf.keras.initializers.Zeros()(shape=(1, dim_counts))

    self.inflated_loc_logits = tf.keras.Variable(inflated_loc_logits,
                                                 trainable=True,
                                                 dtype=tf.float32)
    # <<< initialise mixtures <<<

    # get training variables
    self.t_vars = [*self.layers['mixture_logits'].trainable_variables,
                   *self.layers['encoder_classifier'].trainable_variables,
                   self.logits,
                   self.log_count_spread,
                   self.inflated_loc_logits]


  def get_assignment_distribution(self, logits):
    return tfp.distributions.RelaxedOneHotCategorical(logits=logits, temperature=self.temperature)


  def get_mixture_distributions(self, library_sizes):
    _broad_mix_shape = (1, 1, self.dim_mixtures, self.dim_counts)
    _broad_data_shape = library_sizes.shape + (1, 1)
    _broad_inflation_shape = (1, 1, 1, self.dim_counts)
    
    inflated_loc_probs = tf.math.sigmoid(self.inflated_loc_logits) * .9
    count_spread = tf.nn.softmax(self.log_count_spread, axis=1)

    logits = tf.reshape(self.logits, _broad_mix_shape)
    total_count = tf.reshape(library_sizes, _broad_data_shape) * tf.reshape(count_spread, _broad_mix_shape)
    inflated_loc_probs = tf.reshape(inflated_loc_probs, _broad_inflation_shape)
    
    return tfp.distributions.ZeroInflatedNegativeBinomial(
        logits=logits,
        total_count=total_count,
        inflated_loc_probs=inflated_loc_probs,
        require_integer_total_count=False,
    )


  #def get_mixture_mean(self):
  #  return tf.math.exp(self.log_total_count) * tf.math.exp(-self.logits)


  def decode_log_mixture_probs(self, latent):
    paddings_probs = tf.constant([[0, 0], [1, 0]])
    log_mixture_probs = tf.pad(self.layers['mixture_logits'](latent),
                                paddings_probs,
                                'CONSTANT')
    log_mixture_probs = tf.math.log(tf.nn.softmax(log_mixture_probs, axis=-1))

    # minimum topic proportion is EPS
    log_eps = tf.constant(tf.math.log(self.eps), shape=log_mixture_probs.shape)
    return tf.reduce_logsumexp(tf.stack([log_mixture_probs, log_eps]), axis=0)


  def decode(self, latent, data, sample=False):
    counts, library_sizes = data
    
    log_mixture_probs = self.decode_log_mixture_probs(latent)
    log_mixture_probs = tf.expand_dims(log_mixture_probs, 1) #(n_batch, 1, n_mix)

    mixtures = self.get_mixture_distributions(library_sizes)

    _broad_counts = tf.expand_dims(counts, -2) #(n_batch, n_subbatch, 1, n_counts)
    log_likelihoods = mixtures.log_prob(_broad_counts) 
    log_likelihoods = tf.reduce_sum(log_likelihoods, axis=-1)

    assignment_sample = self.get_assignment_distribution(log_mixture_probs + log_likelihoods).sample() if sample else None

    return assignment_sample, log_likelihoods, log_mixture_probs


  def loss(self,
           data,
           latent,
           encoder_assignment_sample,
           encoder_assignment_logits,
           beta=1):
    counts, library_sizes = data
    _batch_size = counts.shape[0]
             
    _, log_likelihoods, log_mixture_probs = self.decode(latent, data)

    kl_loss = tf.reduce_sum(
          tfp.distributions.Categorical(logits=encoder_assignment_logits).kl_divergence(
              tfp.distributions.Categorical(logits=log_mixture_probs)
              )
    )
    kl_loss /= _batch_size

    ll_loss = -tf.reduce_sum(encoder_assignment_sample * log_likelihoods)
    ll_loss /= _batch_size

    return tf.reduce_sum([self.prop_loss_scale*kl_loss,
                          ll_loss,
                          *self.layers['mixture_logits'].losses,
                          *self.layers['encoder_classifier'].losses])


  def encode(self, data):
    counts, library_sizes = data
    assignment_logits = self.layers['encoder_classifier'](counts)
    assignment_sample = self.get_assignment_distribution(assignment_logits).sample()

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
        'dim_counts':self.dim_counts,
        'head_name':self.head_name,
        'head_type':self.head_type,
        'temperature':self.temperature,
        'prop_loss_scale':self.prop_loss_scale,
        "layer_configs": {key: layer.get_config() for key, layer in self.layers.items()},
        'mixture_params':{
            'logits':self.logits.numpy().tolist(),
            'log_count_spread':self.log_count_spread.numpy().tolist(),
            'inflated_loc_logits':self.inflated_loc_logits.numpy().tolist(),
        },
    }
    return config

  def from_config(config):
    return FACTMx_head_ZINB_mixture2(**config)
