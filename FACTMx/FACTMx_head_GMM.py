"""
Gaussian-mixture data head for FACTMx.

The GMM head models observations as draws from latent Gaussian mixture
components.  It provides both an encoder-side classifier that proposes mixture
assignments for observations and a decoder-side mapping from the shared latent
state to mixture proportions.
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import tensorflow.keras as keras

from FACTMx.FACTMx_head import FACTMx_head



class FACTMx_head_GMM(FACTMx_head):
  """
  Gaussian mixture head for sub-batched continuous observations.
  
  The head learns mixture component parameters, predicts latent-dependent mixture
  proportions, and uses an encoder-side classifier to propose assignments for the
  observed sub-batch.
  """
  head_type = 'GMM'

  def __init__(self,
               dim, dim_latent, dim_normal,
               head_name,
               layer_configs={'mixture_logits':'linear', 'encoder_classifier':'linear'},
               mixture_params={'loc': 'random', 'log_cov_diag': 0., 'cov_perturb_factor': None},
               temperature=1E-4, 
               eps=1E-3, 
               cov_eps=1E-1,
               max_n_perturb_factor=2,):
    """Initialise decoder proportions, encoder classifier, and mixture parameters."""
    super().__init__(dim, dim_latent, head_name)

    self.dim_normal = dim_normal
    self.temperature = temperature
    self.eps = eps
    self.cov_eps = cov_eps
    self.n_cov_perturb_factor = min(dim_normal, max_n_perturb_factor)

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
    # Component means, diagonal covariance terms, and low-rank perturbations are
    # trainable variables rather than Keras layers, so they are added manually to
    # t_vars below.
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
    """Return a relaxed categorical distribution over mixture assignments."""
    return tfp.distributions.RelaxedOneHotCategorical(logits=logits,
                                                      temperature=self.temperature)


  def get_mixture_distributions(self):
    """Build component Normal distributions with diagonal plus low-rank covariance."""
    return tfp.distributions.MultivariateNormalDiagPlusLowRankCovariance(
        self.mixture_locs,
        tf.keras.activations.relu(self.mixture_log_covs) + self.cov_eps,
        self.mixture_cov_perturb
    )

  
  def decode_mixture_logits(self, latent):
    """Predict numerically-stabilized log mixture proportions from latent points."""
    mixture_logits = self.layers['mixture_logits'](latent) 

    # minimum topic proportion is EPS
    log_eps = tf.constant(tf.math.log(self.eps), shape=mixture_logits.shape)
    return tf.reduce_logsumexp(tf.stack([mixture_logits, log_eps]), axis=0)

  
  def decode(self, latent, data, sample=True):
    """Return assignment samples/logits and mixture logits for a batch."""
    mixture_logits = self.decode_mixture_logits(latent)
    mixture_logits = tf.reshape(mixture_logits, (-1, 1, self.dim))

    mixtures = self.get_mixture_distributions()

    log_likelihoods = mixtures.log_prob(tf.expand_dims(data, -2))

    # Posterior assignment logits combine prior mixture proportions with the
    # component likelihood of each observed item.
    assignment_logits = tf.math.add(mixture_logits, log_likelihoods)
    assignment_sample = self.get_assignment_distribution(assignment_logits).sample() if sample else None

    return assignment_sample, assignment_logits, mixture_logits


  def loss(self,
            data,
            latent,
            encoder_assignment_sample,
            encoder_assignment_logits,
            beta=1):
    
    """Compute assignment KL plus expected negative component log-likelihood."""
    assignment_sample, assignment_logits, mixture_logits = FACTMx_head_GMM.decode(self, latent, data, sample=False)

    log_likelihoods = tf.math.subtract(assignment_logits, mixture_logits)

    kl_divergence = tf.reduce_mean(
          tfp.distributions.OneHotCategorical(logits=encoder_assignment_logits).kl_divergence(
              tfp.distributions.OneHotCategorical(logits=mixture_logits)
              )
    )

    log_likelihood = tf.reduce_sum(
        tf.math.multiply(encoder_assignment_sample, log_likelihoods),
    )

    batch_size, subbatch_size, _ = data.shape

    ll_loss = -log_likelihood/batch_size/subbatch_size

    return tf.reduce_sum([kl_divergence,
                          ll_loss,
                          *self.layers['mixture_logits'].losses,
                          *self.layers['encoder_classifier'].losses])


  def encode(self, data):
    """Classify observations into mixture assignments and form encoder log proportions."""
    assignment_logits = self.layers['encoder_classifier'](data)
    assignment_sample = self.get_assignment_distribution(assignment_logits).sample()

    proportions_sample = tf.reduce_mean(assignment_sample, axis=1) + self.eps
    encoder_input = tf.math.log(proportions_sample)

    return {'encoder_input': encoder_input,
            'encoder_assignment_sample': assignment_sample,
            'encoder_assignment_logits': assignment_logits}


  def get_config(self):
    """Serialize GMM dimensions, layer configs, and learned mixture parameters."""
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
    """Reconstruct a GMM head from a config dictionary."""
    return FACTMx_head_GMM(**config)
