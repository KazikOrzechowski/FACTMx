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


def ch_score(X, labels):
  n_topics = labels.shape[-1]
  dim_gmm = X.shape[-1]

  X = tf.reshape(X, (-1, dim_gmm))
  labels = tf.reshape(labels, (-1, n_topics))

  n_samples = X.shape[0]

  total_mean = tf.reduce_mean(X, axis=0)
  
  clusters = tf.expand_dims(X, -2) * tf.expand_dims(labels, -1)
  cluster_pops = tf.reduce_sum(labels, axis=0)
  cluster_means = tf.reduce_sum(clusters, axis=0) / tf.expand_dims(cluster_pops, -1)

  extra_disp = cluster_pops * tf.reduce_sum((cluster_means - tf.expand_dims(total_mean, 0))**2, axis=-1)
  extra_disp = tf.reduce_sum(extra_disp)

  intra_disp = tf.reduce_sum((clusters - tf.expand_dims(cluster_means, 0))**2)

  if intra_disp == 0.0:
    return 1.0
  else:
    return extra_disp * (n_samples - n_topics) / (intra_disp * (n_topics - 1.0))


class FACTMx_head_GMM_prop(FACTMx_head):
  head_type = 'GMM_prop'
  log_mult = False
  ch_scale = 0.0

  def __init__(self,
               dim, dim_latent, dim_normal,
               head_name,
               layer_configs={'mixture_logits':'linear', 'encoder_classifier':'linear'},
               mixture_params={'loc': 'random', 'log_cov_diag': 0., 'cov_perturb_factor': None},
               temperature=1E-4, 
               eps=1E-3, 
               cov_eps=1E-1,
               max_n_perturb_factor=2,
               l1_scale=.1,
               prop_loss_scale=1.,
               regularise_orthogonal=True):
    super().__init__(dim, dim_latent, head_name)

    self.dim_normal = dim_normal
    self.temperature = temperature
    self.eps = eps
    self.cov_eps = cov_eps
    self.n_cov_perturb_factor = min(dim_normal, max_n_perturb_factor)
    self.l1_scale = l1_scale
    self.prop_loss_scale = prop_loss_scale
    self.regularise_orthogonal = regularise_orthogonal

    # >>> initialise layers >>>
    mixture_logits_config = layer_configs.pop('mixture_logits', 'linear')
    if mixture_logits_config == 'linear':
      self.layers['mixture_logits'] = tf.keras.Sequential(
                                        [tf.keras.Input(shape=(self.dim_latent,)),
                                         tf.keras.layers.Dense(units=self.dim,
                                                               kernel_initializer='random_normal',
                                                               activation='log_softmax',
                                                               bias_initializer='ones')]
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
    mixture_locs = mixture_params.pop('loc', 'random')
    if mixture_locs == 'random':
      mixture_locs = tf.keras.initializers.Orthogonal()(shape=(dim, dim_normal))

    self.mixture_locs = tf.keras.Variable(mixture_locs,
                                          trainable=True,
                                          dtype=tf.float32)

    mixture_log_covs = mixture_params.pop('log_cov_diag', 0.)
    if isinstance(mixture_log_covs, float):
      mixture_log_covs = mixture_log_covs + tf.keras.initializers.Zeros()(shape=(dim, dim_normal))
    
    self.mixture_log_covs = tf.keras.Variable(mixture_log_covs,
                                              trainable=True,
                                              dtype=tf.float32)

    mixture_cov_perturb = mixture_params.pop('cov_perturb_factor', None)
    if mixture_cov_perturb is None:
      _cov_perturb_shape = (dim, dim_normal, self.n_cov_perturb_factor)
      mixture_cov_perturb = tf.keras.initializers.RandomNormal()(shape=_cov_perturb_shape)

    self.mixture_cov_perturb = tf.keras.Variable(mixture_cov_perturb,
                                                 trainable=True,
                                                 dtype=tf.float32)
    # <<< initialise mixtures <<<

    # get training variables
    self.t_vars = [*self.layers['mixture_logits'].trainable_variables,
                   *self.layers['encoder_classifier'].trainable_variables,
                   self.mixture_locs,
                   self.mixture_log_covs,
                   self.mixture_cov_perturb]


  def get_assignment_distribution(self, logits):
    return tfp.distributions.RelaxedOneHotCategorical(logits=logits,
                                                        temperature=self.temperature)


  def get_mixture_distributions(self):
    return tfp.distributions.MultivariateNormalDiagPlusLowRankCovariance(
        self.mixture_locs,
        tf.keras.activations.relu(self.mixture_log_covs) + self.cov_eps,
        self.mixture_cov_perturb
    )

  
  def decode_mixture_logits(self, latent):
    mixture_logits = self.layers['mixture_logits'](latent) 

    # minimum topic proportion is EPS
    log_eps = tf.constant(tf.math.log(self.eps), shape=mixture_logits.shape)
    return tf.reduce_logsumexp(tf.stack([mixture_logits, log_eps]), axis=0)

  
  def decode(self, latent, data, sample=True):
    mixture_logits = self.decode_mixture_logits(latent)
    mixture_logits = tf.reshape(mixture_logits, (-1, 1, self.dim))

    mixtures = self.get_mixture_distributions()

    log_likelihoods = mixtures.log_prob(tf.expand_dims(data, -2))

    assignment_logits = tf.math.add(mixture_logits, log_likelihoods)
    assignment_sample = self.get_assignment_distribution(assignment_logits).sample() if sample else None

    return assignment_sample, assignment_logits, mixture_logits

# last working
  # def loss(self,
  #           data,
  #           latent,
  #           encoder_assignment_sample,
  #           encoder_assignment_logits,
  #           beta=1):
  #   _, assignment_logits, mixture_logits = FACTMx_head_GMM_prop.decode(self, latent, data, sample=False)

  #   log_likelihoods = tf.math.subtract(assignment_logits, mixture_logits)

  #   kl_divergence = tf.reduce_sum(
  #         tfp.distributions.OneHotCategorical(logits=encoder_assignment_logits).kl_divergence(
  #             tfp.distributions.OneHotCategorical(logits=mixture_logits)
  #             )
  #   )
    
  #   #alternate sampling and marginal loss
  #   # if np.random.choice([True, False]):
  #   #   encoder_assignment_sample = tf.math.softmax(encoder_assignment_logits, axis=-1)
  #   log_likelihood = tf.reduce_sum(
  #       tf.math.multiply(encoder_assignment_sample, log_likelihoods),
  #       #axis=2
  #   )
  #   #log_likelihood = tf.reduce_sum(log_likelihood)
  #   cov_matrix = self.get_mixture_distributions().covariance()
  #   mixture_params_penalty = self.l1_scale * (tf.reduce_mean(tf.math.log(cov_matrix ** 2 + 1E-10)) + 
  #                                             tf.reduce_mean(tf.math.exp(cov_matrix)))
  #   if self.regularise_orthogonal:
  #     normalized_topic = tf.math.l2_normalize(self.mixture_locs, axis=0)
  #     mixture_params_penalty += self.l1_scale * tf.reduce_sum(normalized_topic @ tf.transpose(normalized_topic))
  #   batch_size = data.shape[0]

  #   return tf.reduce_sum([self.prop_loss_scale*kl_divergence/batch_size,
  #                         -log_likelihood/batch_size,
  #                         mixture_params_penalty,
  #                         *self.layers['mixture_logits'].losses,
  #                         *self.layers['encoder_classifier'].losses])

  def loss(self,
            data,
            latent,
            encoder_assignment_sample,
            encoder_assignment_logits,
            beta=1):
    
    assignment_sample, assignment_logits, mixture_logits = FACTMx_head_GMM_prop.decode(self, latent, data, sample=self.ch_scale > 0)

    log_likelihoods = tf.math.subtract(assignment_logits, mixture_logits)

    # encoder_probs = tf.math.softmax(encoder_assignment_logits, axis=-1)
    # mean_probs = tf.reduce_mean(encoder_probs, axis=1, keepdims=True) + 1E-50
    kl_divergence = tf.reduce_mean(
          tfp.distributions.OneHotCategorical(logits=encoder_assignment_logits).kl_divergence(
              tfp.distributions.OneHotCategorical(logits=mixture_logits)
              )
    )
    
    #alternate sampling and marginal loss
    # if np.random.choice([True, False]):
    #   encoder_assignment_sample = tf.math.softmax(encoder_assignment_logits, axis=-1)
    log_likelihood = tf.reduce_sum(
        tf.math.multiply(encoder_assignment_sample, log_likelihoods),
        #axis=2
    )
    #log_likelihood = tf.reduce_sum(log_likelihood)
    cov_matrix = self.get_mixture_distributions().covariance()
    diag_cov = tf.linalg.diag_part(cov_matrix)
    diag_cov = tf.reduce_mean(diag_cov ** 2, axis=-1)
    diag_cov /= tf.reduce_sum(diag_cov)
    mixture_params_penalty = self.l1_scale * tf.reduce_mean(diag_cov * tf.math.log(diag_cov)) #we want big entropy of components variances
              
    if self.regularise_orthogonal:
      normalized_topic = tf.math.l2_normalize(self.mixture_locs, axis=0)
      mixture_params_penalty += self.l1_scale * tf.reduce_sum(normalized_topic @ tf.transpose(normalized_topic))
    batch_size, subbatch_size, _ = data.shape #removed subbatch_size for test

    ll_loss = -log_likelihood/batch_size
    if self.log_mult:
      ll_loss /= subbatch_size
    
    ch_loss = -self.ch_loss * ch_score(data, assignment_sample) if self.ch_loss > 0 else 0.
  

    return tf.reduce_sum([self.prop_loss_scale*kl_divergence,
                          ll_loss,
                          ch_loss,
                          mixture_params_penalty,
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
        'dim_normal':self.dim_normal,
        'head_name':self.head_name,
        'head_type':self.head_type,
        'temperature':self.temperature,
        'eps':self.eps,
        'cov_eps':self.cov_eps,
        'max_n_perturb_factor':self.n_cov_perturb_factor,
        "layer_configs": {key: layer.get_config() for key, layer in self.layers.items()},
        'mixture_params':{
            'loc':self.mixture_locs.numpy().tolist(),
            'log_cov_diag':self.mixture_log_covs.numpy().tolist(),
            'cov_perturb_factor':self.mixture_cov_perturb.numpy().tolist()
        },
    }
    return config

  def from_config(config):
    return FACTMx_head_GMM_prop(**config)
