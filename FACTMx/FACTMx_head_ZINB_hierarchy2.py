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


class FACTMx_head_ZINB_hierarchy2(FACTMx_head):
  head_type = 'ZINB_hierarchy2'

  def __init__(self,
               dim, dim_latent, dim_counts, dim_topics,
               head_name,
               dim_levels=1,
               unfrozen_levels=None,
               layer_configs={'mixture_logits':'linear', 'encoder_classifier':'linear'},
               mixture_params_list=None,
               temperature=1E-4,
               eps=1E-3,
               prop_loss_scale=1.):
    super().__init__(dim, dim_latent, head_name)

    self.dim_topics = dim_topics
    self.dim_counts = dim_counts
    self.dim_levels = dim_levels
    self.unfrozen_levels = dim_levels if unfrozen_levels is None else unfrozen_levels
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
                                             tf.keras.layers.Dense(units=self.dim,
                                                                   activation='log_softmax')]
                                          )
    else:
      self.layers['encoder_classifier'] = tf.keras.Sequential.from_config(encoder_classifier_config)

    assert self.layers['encoder_classifier'].input_shape == (None, None, self.dim_counts)
    assert self.layers['encoder_classifier'].output_shape == (None, None, self.dim)
    # <<< initialise layers <<<

    # >>> initialise mixtures >>>
    if mixture_params_list is None:
      mixture_params_list = [dict()] * self.dim_levels
      
    assert self.dim_levels == len(mixture_params_list)

    self.level_logits = []
    self.level_log_total_count = []
    self.level_inflated_loc_logits = []

    for level, level_params in enumerate(mixture_params_list):
      _level_shape = (dim_topics,) * (level+1) + (dim_counts,)
      _logit_init = tf.keras.initializers.RandomNormal(stddev=dim_topics ** (-3*level))
      _default_init = tf.keras.initializers.Zeros()

      if 'logits' in level_params.keys():
        logits = level_params['logits']
        logits_were_init = False
      else:
        logits = _logit_init(shape=_level_shape)
        logits_were_init = True
      if logits_were_init and level > 0:
        logits += tf.expand_dims(self.level_logits[-1], axis=-1)
      
      self.level_logits.append(tf.keras.Variable(logits,
                                                 trainable=True,
                                                 dtype=tf.float32))

      log_total_count = level_params.pop('log_total_count', _default_init(shape=_level_shape))
      self.level_log_total_count.append(tf.keras.Variable(log_total_count,
                                                          trainable=True,
                                                          dtype=tf.float32))

      inflated_loc_logits = level_params.pop('inflated_loc_logits', _default_init(shape=_level_shape))
      self.level_inflated_loc_logits.append(tf.keras.Variable(inflated_loc_logits,
                                                              trainable=True,
                                                              dtype=tf.float32))
    # <<< initialise mixtures <<<

    # get training variables
    self.t_vars = [*self.layers['mixture_logits'].trainable_variables,
                   *self.layers['encoder_classifier'].trainable_variables,
                   *self.level_logits,
                   *self.level_log_total_count,
                   *self.level_inflated_loc_logits]


  def get_assignment_distribution(self, logits):
    return tfp.distributions.RelaxedOneHotCategorical(logits=logits, temperature=self.temperature)


  def decode_log_mixture_probs(self, latent):
    log_mixture_probs = self.layers['mixture_logits'](latent)

    # minimum topic proportion is EPS
    log_eps = tf.constant(tf.math.log(self.eps), shape=log_mixture_probs.shape)
    return tf.reduce_logsumexp(tf.stack([log_mixture_probs, log_eps]), axis=0)


  def get_mixture_distributions(self, level):
    logits = self.level_logits[level]
    log_total_count = self.level_log_total_count[level]
    inflated_loc_logits = self.level_inflated_loc_logits[level]
    
    probs = tf.math.sigmoid(logits) + self.eps
    total_count = tf.math.exp(log_total_count) + self.eps
    inflated_loc_probs = tf.math.sigmoid(inflated_loc_logits) * 0.75
    #reshape to flat mixtures
    _flat_shape = (-1, self.dim_counts)

    return tfp.distributions.ZeroInflatedNegativeBinomial(
        probs=tf.reshape(probs, _flat_shape),
        total_count=tf.reshape(total_count, _flat_shape),
        inflated_loc_probs=tf.reshape(inflated_loc_probs, _flat_shape),
        require_integer_total_count=False,
    )


  def decode(self, latent, data):
    log_mixture_probs = self.decode_log_mixture_probs(latent)

    level_assignment_loglikelihoods = []
    for level in range(self.unfrozen_levels):
      mixtures = self.get_mixture_distributions(level)

      log_likelihood = mixtures.log_prob(tf.expand_dims(data, 2))
      log_likelihood = tf.reduce_sum(log_likelihood, axis=-1)

      level_assignment_loglikelihoods.append(log_likelihood)

    return level_assignment_loglikelihoods, log_mixture_probs


  def loss(self,
            data,
            latent,
            encoder_assignment_sample,
            encoder_assignment_logits,
            beta=1):
    _batch_size, _subbatch_size, _ = data.shape
    level_assignment_loglikelihoods, log_mixture_probs = self.decode(latent, data)

    log_mixture_probs = tf.expand_dims(log_mixture_probs, 1)
    kl_divergence = tf.reduce_sum(
          tfp.distributions.Categorical(logits=encoder_assignment_logits).kl_divergence(
              tfp.distributions.Categorical(logits=log_mixture_probs)
              )
    )

    level_loglikelihoods = []
    for level in range(self.dim_levels-1, -1, -1):
      if level < self.unfrozen_levels:
        log_likelihood = tf.reduce_sum(
            tf.math.multiply(encoder_assignment_sample, level_assignment_loglikelihoods[level]),
        )
  
        level_loglikelihoods.append(log_likelihood)

      encoder_assignment_sample = tf.reshape(encoder_assignment_sample, (_batch_size, _subbatch_size, -1, self.dim_topics))
      encoder_assignment_sample = tf.reduce_sum(encoder_assignment_sample, axis=-1)

    kl_loss = kl_divergence * self.prop_loss_scale / _batch_size
    ll_loss = -tf.reduce_sum(level_loglikelihoods) / _batch_size

    return tf.reduce_sum([kl_loss,
                          ll_loss,
                          *self.layers['mixture_logits'].losses,
                          *self.layers['encoder_classifier'].losses])


  def encode(self, data):
    assignment_logits = self.layers['encoder_classifier'](data)
    assignment_logits = tf.math.log(tf.math.exp(assignment_logits) + self.eps)
    assignment_sample = self.get_assignment_distribution(assignment_logits).sample()

    proportions_sample = tf.reduce_mean(assignment_sample, axis=1) + self.eps
    log_proportions_sample = tf.math.log(proportions_sample)

    return {'encoder_input': log_proportions_sample,
            'encoder_assignment_sample': assignment_sample,
            'encoder_assignment_logits': assignment_logits}


  def get_config(self):
    config = {
        'dim':self.dim,
        'dim_latent':self.dim_latent,
        'dim_counts':self.dim_counts,
        'dim_topics':self.dim_topics,
        'dim_levels':self.dim_levels,
        'unfrozen_levels':self.unfrozen_levels,
        'head_name':self.head_name,
        'head_type':self.head_type,
        'temperature':self.temperature,
        'prop_loss_scale':self.prop_loss_scale,
        "layer_configs": {key: layer.get_config() for key, layer in self.layers.items()},
        'mixture_params_list':[
          {
          'logits':logits.numpy().tolist(),
          'log_total_count':log_total_count.numpy().tolist(),
          'inflated_loc_logits':inflated_loc_logits.numpy().tolist(),
          } for logits, log_total_count, inflated_loc_logits in zip(self.level_logits, 
                                                                    self.level_log_total_count, 
                                                                    self.level_inflated_loc_logits)
        ],
    }
    return config

  def from_config(config):
    return FACTMx_head_ZINB_hierarchy2(**config)
