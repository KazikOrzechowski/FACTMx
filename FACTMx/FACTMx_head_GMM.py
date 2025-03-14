import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from FACTMx.FACTMx_head import FACTMx_head


class FACTMx_head_GMM(FACTMx_head):
  head_type = 'GMM'

  def __init__(self,
               dim, dim_latent, dim_normal,
               head_name,
               decode_config='linear',
               encoder_classifier_config='linear',
               mixture_params={'loc': 'random', 'log_cov_diag': 0., 'cov_perturb_factor': None},
               temperature=1E-4, eps=1E-3, max_n_perturb_factor=2):
    super().__init__(dim, dim_latent, head_name)
    self.dim_normal = dim_normal
    self.temperature = temperature
    self.eps = eps
    self.n_cov_perturb_factor = min(dim_normal, max_n_perturb_factor)

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

    if encoder_classifier_config == 'linear':
      self.encoder_classifier = tf.keras.Sequential(
                                  [tf.keras.Input(shape=(None, self.dim_normal)),
                                   tf.keras.layers.Dense(units=self.dim+1,
                                                         activation='log_softmax')]
                                )
    else:
      self.encoder_classifier = tf.keras.Sequential.from_config(encoder_classifier_config)

    assert self.encoder_classifier.input_shape == (None, None, self.dim_normal)
    assert self.encoder_classifier.output_shape == (None, None, self.dim+1)

    mixture_locs = mixture_params.pop('loc', 'random')
    if mixture_locs == 'random':
      mixture_locs = tf.keras.initializers.Orthogonal()(shape=(dim+1, dim_normal))

    self.mixture_locs = tf.keras.Variable(mixture_locs,
                                          trainable=True,
                                          dtype=tf.float32)

    mixture_log_covs = mixture_params.pop('log_cov_diag', 0.)
    if isinstance(mixture_log_covs, float):
      mixture_log_covs = mixture_log_covs + tf.keras.initializers.Zeros()(shape=(dim+1, dim_normal))

    self.mixture_log_covs = tf.keras.Variable(mixture_log_covs,
                                              trainable=True,
                                              dtype=tf.float32)

    mixture_cov_perturb = mixture_params.pop('cov_perturb_factor', None)
    if mixture_cov_perturb is None:
      _cov_perturb_shape = (dim+1, dim_normal, self.n_cov_perturb_factor)
      mixture_cov_perturb = tf.keras.initializers.Zeros()(shape=_cov_perturb_shape)

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
    paddings_probs = tf.constant([[0, 0], [1, 0]])
    log_mixture_probs = tf.pad(self.decode_model(latent),
                                paddings_probs,
                                'CONSTANT')
    log_mixture_probs = tf.keras.activations.log_softmax(log_mixture_probs,
                                                          axis=-1)
    return log_mixture_probs


  def decode(self, latent, data):
    log_mixture_probs = self.decode_log_mixture_probs(latent)
    log_mixture_probs = tf.reshape(log_mixture_probs,
                                    (-1, 1, self.dim+1))

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

    kl_divergence = tf.reduce_mean(
          tfp.distributions.OneHotCategorical(logits=encoder_assignment_logits).kl_divergence(
              tfp.distributions.OneHotCategorical(logits=log_mixture_probs)
              )
    )

    log_likelihood = tf.reduce_sum(
        tf.math.multiply(encoder_assignment_sample, log_likelihoods),
        axis=2
    )
    log_likelihood = tf.reduce_mean(log_likelihood)

    return tf.reduce_sum([beta*kl_divergence,
                          -log_likelihood,
                          *self.decode_model.losses])


  def encode(self, data):
    assignment_logits = self.encoder_classifier(data)
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
        'decode_config':self.decode_model.get_config()
    }
    return config

  def from_config(config):
    return FACTMx_head_GMM(**config)

  def save_weights(self, head_path):
    self.decode_model.save_weights(f'{head_path}_decode_model.weights.h5')
    self.encoder_classifier.save_weights(f'{head_path}_encoder_classifier.weights.h5')

  def load_weights(self, head_path):
    self.decode_model.load_weights(f'{head_path}_decode_model.weights.h5')
    self.encoder_classifier.load_weights(f'{head_path}_encoder_classifier.weights.h5')
