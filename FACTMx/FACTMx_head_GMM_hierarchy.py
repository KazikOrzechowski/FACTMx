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


class FACTMx_head_GMM_hierarchy(FACTMx_head):
  head_type = 'GMM_hierarchy'

  def __init__(self,
               dim, dim_latent, dim_normal, dim_topics,
               head_name,
               dim_levels=1,
               unfrozen_levels=None,
               layer_configs={'mixture_logits':'linear', 'encoder_classifier':'linear'},
               mixture_params_list=None,
               dim_cov_perturb=2,
               temperature=1E-4,
               eps=1E-3,
               prop_loss_scale=1.):
    super().__init__(dim, dim_latent, head_name)

    self.dim_topics = dim_topics
    self.dim_normal = dim_normal
    self.dim_levels = dim_levels
    self.unfrozen_levels = dim_levels if unfrozen_levels is None else unfrozen_levels
    self.dim_cov_perturb = min(dim_cov_perturb, dim_normal)
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
                                            [tf.keras.Input(shape=(None, self.dim_normal)),
                                             tf.keras.layers.Dense(units=self.dim,
                                                                   activation='log_softmax')]
                                          )
    else:
      self.layers['encoder_classifier'] = tf.keras.Sequential.from_config(encoder_classifier_config)

    assert self.layers['encoder_classifier'].input_shape == (None, None, self.dim_normal)
    assert self.layers['encoder_classifier'].output_shape == (None, None, self.dim)
    # <<< initialise layers <<<

    # >>> initialise mixtures >>>
    if mixture_params_list is None:
      mixture_params_list = [dict()] * self.dim_levels
      
    assert self.dim_levels == len(mixture_params_list)

    self.level_locs = []
    self.level_log_scales = []
    self.level_cov_perturbs = []

    for level, level_params in enumerate(mixture_params_list):
      _level_shape = (dim_topics,) * (level+1) + (dim_normal,)
      _level_perturb_shape = _level_shape + (dim_cov_perturb,)
      
      _loc_init = tf.keras.initializers.Orthogonal(gain=dim_normal * dim_topics ** (-level-1.5))(shape=_level_shape)
      _log_scale_init = tf.keras.initializers.Ones()(_level_shape) * (-level)
      _perturb_init = tf.keras.initializers.RandomUniform(-1, 1)(_level_perturb_shape) * np.exp(-level-1)
      
      if 'loc' in level_params.keys():
        loc = level_params['loc']
        loc_was_init = False
      else:
        loc = _loc_init
        loc_was_init = True
      if loc_was_init and level > 0:
        loc += tf.expand_dims(self.level_locs[-1], axis=-2)
      
      self.level_locs.append(tf.keras.Variable(loc,
                                               name=f'{head_name}_locs_{level}',
                                               trainable=True,
                                               dtype=tf.float32))

      log_scale = level_params.pop('log_scale', _log_scale_init)
      self.level_log_scales.append(tf.keras.Variable(log_scale,
                                                     name=f'{head_name}_logscales_{level}',
                                                     trainable=True,
                                                     dtype=tf.float32))

      cov_perturb = level_params.pop('cov_perturb', _perturb_init)
      self.level_cov_perturbs.append(tf.keras.Variable(cov_perturb,
                                                       name=f'{head_name}_covperturbs_{level}',
                                                       trainable=True,
                                                       dtype=tf.float32))
    # <<< initialise mixtures <<<

    # get training variables
    self.t_vars = [*self.layers['mixture_logits'].trainable_variables,
                   *self.layers['encoder_classifier'].trainable_variables,
                   *self.level_locs,
                   *self.level_log_scales,
                   *self.level_cov_perturbs]


  def get_assignment_distribution(self, logits):
    return tfp.distributions.RelaxedOneHotCategorical(logits=logits, temperature=self.temperature)


  def decode_log_mixture_probs(self, latent):
    log_mixture_probs = self.layers['mixture_logits'](latent)

    # minimum topic proportion is EPS
    log_eps = tf.constant(tf.math.log(self.eps), shape=log_mixture_probs.shape)
    return tf.reduce_logsumexp(tf.stack([log_mixture_probs, log_eps]), axis=0)


  def get_mixture_distributions(self, level):
    #reshape to flat mixtures
    _flat_shape = (-1, self.dim_normal)

    loc = self.level_locs[level]
    loc = tf.reshape(loc, _flat_shape)

    log_scale = self.level_log_scales[level]
    log_scale = tf.reshape(log_scale, _flat_shape)
    cov_diag_factor = tf.math.exp(log_scale) + self.eps

    cov_perturb_factor = self.level_cov_perturbs[level]
    cov_perturb_factor = tf.reshape(cov_perturb_factor, (-1, self.dim_normal, self.dim_cov_perturb))

    return tfp.distributions.MultivariateNormalDiagPlusLowRankCovariance(
        loc,
        cov_diag_factor,
        cov_perturb_factor
    )

  def decode(self, latent, data):
    log_mixture_probs = self.decode_log_mixture_probs(latent)

    level_assignment_loglikelihoods = []
    for level in range(self.unfrozen_levels):
      mixtures = self.get_mixture_distributions(level)

      log_likelihood = mixtures.log_prob(tf.expand_dims(data, 2))
      
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
        'dim_normal':self.dim_normal,
        'dim_topics':self.dim_topics,
        'dim_levels':self.dim_levels,
        'dim_cov_perturb':self.dim_cov_perturb,
        'unfrozen_levels':self.unfrozen_levels,
        'head_name':self.head_name,
        'head_type':self.head_type,
        'temperature':self.temperature,
        'prop_loss_scale':self.prop_loss_scale,
        "layer_configs": {key: layer.get_config() for key, layer in self.layers.items()},
        'mixture_params_list':[
          {
          'loc':loc.numpy().tolist(),
          'log_scale':log_scale.numpy().tolist(),
          'cov_perturb':cov_perturb.numpy().tolist()
          } for loc, log_scale, cov_perturb in zip(self.level_locs,
                                                   self.level_log_scales,
                                                   self.level_cov_perturbs)
        ],
    }
    return config

  def from_config(config):
    return FACTMx_head_GMM_hierarchy(**config)
