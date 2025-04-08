import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from FACTMx.FACTMx_head import FACTMx_head


class FACTMx_head_GMM_prop(FACTMx_head):
  head_type = 'GMM_prop'

  def __init__(self,
               dim, dim_latent, dim_normal,
               head_name,
               decode_mixture_config='linear',
               encoder_classifier_config='linear',
               mixture_params={'loc': 'random', 'log_cov_diag': 0., 'cov_perturb_factor': None},
               temperature=1E-4, 
               eps=1E-3, 
               max_n_perturb_factor=2,
               l1_scale=.1,
               prop_loss_scale=1.,
               regularise_orthogonal=True):
    super().__init__(dim, dim_latent, head_name)
    self.dim_normal = dim_normal
    self.temperature = temperature
    self.eps = eps
    self.n_cov_perturb_factor = min(dim_normal, max_n_perturb_factor)
    self.l1_scale = l1_scale
    self.prop_loss_scale = prop_loss_scale
    self.regularise_orthogonal = regularise_orthogonal

    if decode_mixture_config == 'linear':
      self.decode_model = tf.keras.Sequential(
                            [tf.keras.Input(shape=(self.dim_latent,)),
                             tf.keras.layers.Dense(units=self.dim,
                                                   kernel_initializer='orthogonal',
                                                   activation='sigmoid')]
                          )
    else:
      self.decode_model = tf.keras.Sequential.from_config(decode_mixture_config)

    assert self.decode_model.output_shape == (None, self.dim)
    assert self.decode_model.input_shape == (None, self.dim_latent)

    if encoder_classifier_config == 'linear':
      self.encoder_classifier = tf.keras.Sequential(
                                  [tf.keras.Input(shape=(None, self.dim_normal)),
                                   tf.keras.layers.Dense(units=self.dim,
                                                         activation='log_softmax')]
                                )
    else:
      self.encoder_classifier = tf.keras.Sequential.from_config(encoder_classifier_config)

    assert self.encoder_classifier.input_shape == (None, None, self.dim_normal)
    assert self.encoder_classifier.output_shape == (None, None, self.dim)

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
      mixture_cov_perturb = tf.keras.initializers.Zeros()(shape=_cov_perturb_shape)
    elif mixture_cov_perturb == 'random':
      _cov_perturb_shape = (dim, dim_normal, self.n_cov_perturb_factor)
      mixture_cov_perturb = tf.keras.initializers.Orthogonal()(shape=_cov_perturb_shape)

    self.mixture_cov_perturb = tf.keras.Variable(mixture_cov_perturb,
                                                 trainable=True,
                                                 dtype=tf.float32)

    self.t_vars = [*self.decode_model.trainable_variables,
                   *self.encoder_classifier.trainable_variables,
                   self.mixture_locs,
                   self.mixture_log_covs,
                   self.mixture_cov_perturb]


  def get_assignment_distribution(self, logits):
    return tfp.distributions.RelaxedOneHotCategorical(logits=logits,
                                                        temperature=self.temperature)


  def get_mixture_distributions(self):
    return tfp.distributions.MultivariateNormalDiagPlusLowRankCovariance(
        self.mixture_locs,
        tf.keras.activations.relu(self.mixture_log_covs) + self.eps,
        self.mixture_cov_perturb
    )


  def decode_log_mixture_probs(self, latent):
    mixture_probs = self.decode_model(latent)
    
    # minimum topic proportion is EPS
    eps = tf.constant(self.eps, shape=mixture_probs.shape)
    return tf.math.log(mixture_probs + eps)

  
  def decode(self, latent, data):
    log_mixture_probs = self.decode_log_mixture_probs(latent)
    log_mixture_probs = tf.reshape(log_mixture_probs,
                                    (-1, 1, self.dim))

    mixtures = self.get_mixture_distributions()

    _broadcastable_shape = (-1, tf.shape(data)[1], 1, self.dim_normal)
    log_likelihoods = mixtures.log_prob(tf.reshape(data, _broadcastable_shape))

    assignment_logits = tf.math.add(log_mixture_probs, log_likelihoods)
    assignment_sample = self.get_assignment_distribution(assignment_logits).sample()

    return assignment_sample, assignment_logits, log_mixture_probs


  def loss(self,
            data,
            latent,
            encoder_assignment_sample,
            encoder_assignment_logits,
            beta=1):
    _, assignment_logits, log_mixture_probs = self.decode(latent, data)

    log_likelihoods = tf.math.subtract(assignment_logits, log_mixture_probs)

    kl_divergence = tf.reduce_sum(
          tfp.distributions.OneHotCategorical(logits=encoder_assignment_logits).kl_divergence(
              tfp.distributions.OneHotCategorical(logits=log_mixture_probs)
              )
    )

    log_likelihood = tf.reduce_sum(
        tf.math.multiply(encoder_assignment_sample, log_likelihoods),
        #axis=2
    )
    #log_likelihood = tf.reduce_sum(log_likelihood)

    mixture_params_penalty = self.l1_scale * (tf.reduce_sum(tf.math.abs(self.mixture_cov_perturb)) + 
                                              tf.reduce_sum(tf.math.exp(self.mixture_log_covs)))
    if self.regularise_orthogonal:
      normalized_topic = tf.math.l2_normalize(self.mixture_locs, axis=0)
      mixture_params_penalty += self.l1_scale * tf.reduce_sum(normalized_topic @ tf.transpose(normalized_topic))
    batch_size = data.shape[0]

    return tf.reduce_sum([self.prop_loss_scale*kl_divergence/batch_size,
                          -log_likelihood/batch_size,
                          mixture_params_penalty,
                          *self.decode_model.losses])


  def encode(self, data):
    assignment_logits = self.encoder_classifier(data)
    assignment_sample = self.get_assignment_distribution(assignment_logits).sample()

    proportions_sample = tf.reduce_mean(assignment_sample, axis=1) + self.eps

    return {'encoder_input': proportions_sample,
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
        'max_n_perturb_factor':self.n_cov_perturb_factor,
        'mixture_params':{
            'loc':self.mixture_locs.numpy().tolist(),
            'log_cov_diag':self.mixture_log_covs.numpy().tolist(),
            'cov_perturb_factor':self.mixture_cov_perturb.numpy().tolist()
        },
        'encoder_classifier_config':self.encoder_classifier.get_config(),
        'decode_mixture_config':self.decode_model.get_config()
    }
    return config

  def from_config(config):
    return FACTMx_head_GMM_prop(**config)

  def save_weights(self, head_path):
    self.decode_model.save_weights(f'{head_path}_decode_model.weights.h5')
    self.encoder_classifier.save_weights(f'{head_path}_encoder_classifier.weights.h5')

  def load_weights(self, head_path):
    self.decode_model.load_weights(f'{head_path}_decode_model.weights.h5')
    self.encoder_classifier.load_weights(f'{head_path}_encoder_classifier.weights.h5')
