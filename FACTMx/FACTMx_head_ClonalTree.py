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


class FACTMx_head_ClonalTree(FACTMx_head):
  head_type = 'ClonalTree'

  def __init__(self,
               dim, dim_latent, dim_pos, dim_clones, levels,
               head_name,
               log_mut_assignment=None,
               layer_configs={'log_probs':'linear', 'encoder_classifier':'linear'},
               prob_obs=.5, prob_unobs=.01,
               temperature=1E-4, eps=1E-3,
               prop_loss_scale=1.):
    super().__init__(dim, dim_latent, head_name)
    # dim clones should be tumour leaf clones +1 for the normal clone
    self.levels = levels
    self.dim_clones = dim_clones
    self.dim_pos = dim_pos
    self.prob_obs = prob_obs
    self.prob_unobs = prob_unobs
    self.temperature = temperature
    self.eps = eps
    self.prop_loss_scale = prop_loss_scale

    # >>> initialise layers >>>
    assert self.dim == self.dim_clones + 1 #we classify among tumour clones and one reference clone

    mixture_logits_config = layer_configs.pop('log_probs', 'linear')
    if mixture_logits_config == 'linear':
      self.layers['log_probs'] = tf.keras.Sequential(
                                        [tf.keras.Input(shape=(self.dim_latent,)),
                                         tf.keras.layers.Dense(units=self.dim,
                                                               kernel_initializer='orthogonal')]
                                      )
    else:
      self.layers['log_probs'] = tf.keras.Sequential.from_config(mixture_logits_config)

    assert self.layers['log_probs'].output_shape == (None, self.dim)
    assert self.layers['log_probs'].input_shape == (None, self.dim_latent)

    encoder_classifier_config = layer_configs.pop('encoder_classifier', 'linear')
    if encoder_classifier_config == 'linear':
      self.layers['encoder_classifier'] = tf.keras.Sequential(
                                            [tf.keras.Input(shape=(self.dim_pos,)),
                                             tf.keras.layers.Dense(units=self.dim,
                                                                   activation='log_softmax')]
                                          )
    else:
      self.layers['encoder_classifier'] = tf.keras.Sequential.from_config(encoder_classifier_config)

    assert self.layers['encoder_classifier'].input_shape == (None, self.dim_pos)
    assert self.layers['encoder_classifier'].output_shape == (None, self.dim)
    # <<< initialise layers <<<

    # >>> initialise clones >>>
    self.all_tumour_clones = int(2 ** (self.levels+1) - 1)


    self.log_mut_assignment = tf.Variable(tf.zeros((self.dim_pos, self.all_tumour_clones)),
                                           trainable=True,
                                           dtype=tf.float32)
    self.level_shapes = [(self.dim_pos,) + (2,) * i + (1,) * (self.levels-i) for i in range(self.levels+1)]
    self.level_inds = [slice(2**i - 1, 2**(i+1)-1) for i in range(self.levels+1)]
    # <<< initialise clones <<<

    # get training variables
    self.t_vars = [*self.layers['mixture_logits'].trainable_variables,
                   *self.layers['encoder_classifier'].trainable_variables,
                   *self.log_mut_assignment]

  def get_assignment_distribution(self, logits):
    return tfp.distributions.RelaxedOneHotCategorical(logits=logits, temperature=self.temperature)

  def decode_log_probs(self, latent):
    log_probs = self.layers['log_probs'](latent)

    # minimum topic proportion is EPS
    #log_eps = tf.constant(tf.math.log(self.eps), shape=log_mixture_probs.shape)
    return log_probs

  def get_clone_profiles_sample(self):
    mut_assignment_sample = self.get_assignment_distribution(self.log_mut_assignment).sample()
    all_levels = [tf.reshape(mut_assignment_sample[:,inds], shape) for shape, inds in zip(self.level_shape, self.level_inds)]
    
    clonal_profiles = tf.reduce_sum(tf.stack(all_levels), axis=0)
    clonal_profiles = tf.reshape(clonal_profiles, (self.dim_pos, self.dim_clones))
    clonal_profiles = tf.concat([tf.zeros((self.dim_pos,1)), clonal_profiles]) #add reference clone
    clonal_profiles = tf.transpose(clonal_profiles)

    return clonal_profiles

  def get_clone_distributions(self, clonal_profiles, counts):
    probs = clonal_profiles * self.prob_obs + (1-clonal_profiles) * self.prob_unobs
    return tfp.distributions.Binomial(total_count=counts, probs=probs)

  def decode(self, latent, data):
    counts, mut = data

    log_probs = self.decode_log_probs(latent)
    logits = log_probs + self.get_clone_distributions(self.get_clone_profiles_sample(), counts).log_prob(mut)

    sample = self.get_assignment_distribution(logits).sample()

    return sample, logits, log_probs

  def loss(self,
            data,
            latent,
            encoder_assignment_sample,
            beta=1):
    _batch_size = data[0].shape[0]
    sample, logits, log_probs = self.decode(latent, data)

    log_loss = tf.reduce_sum(encoder_assignment_sample * logits)
    log_loss = log_loss / _batch_size

    return tf.reduce_sum([log_loss,
                          *self.layers['mixture_logits'].losses,
                          *self.layers['encoder_classifier'].losses])

  def encode(self, data):
    assignment_logits = self.layers['encoder_classifier'](data)
    #assignment_logits = tf.math.log(tf.math.exp(assignment_logits) + self.eps)
    assignment_sample = self.get_assignment_distribution(assignment_logits).sample()

    return {'encoder_input': assignment_logits,
            'encoder_assignment_sample': assignment_sample}

  def get_config(self):
    config = {
        'head_name':self.head_name,
        'head_type':self.head_type,
        'dim':self.dim,
        'dim_latent':self.dim_latent,
        'dim_pos':self.dim_pos,
        'dim_clones':self.dim_clones,
        "levels": self.levels,
        "prob_obs": self.prob_obs,
        "prob_unobs": self.prob_unobs,
        'temperature':self.temperature,
        'prop_loss_scale':self.prop_loss_scale,
        "layer_configs": {key: layer.get_config() for key, layer in self.layers.items()},
        "log_mut_assignment": self.log_mut_assignment.numpy().tolist(),
    }
    return config

  def from_config(config):
    return FACTMx_head_ClonalTree(**config)
