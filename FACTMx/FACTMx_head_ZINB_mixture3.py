import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from FACTMx.FACTMx_head import FACTMx_head


class FACTMx_head_ZINB_mixture3(FACTMx_head):
  head_type = 'ZINB_mixture3'

  def __init__(self,
               dim, dim_latent, dim_counts,
               head_name,
               layer_configs={'mixture_logits':'linear', 'encoder_classifier':'linear'},
               mixture_params={'logits':None, 'log_total_count':None, 'inflated_loc_logits':None},
               marker_groups=None,
               temperature=1E-4,
               eps=1E-3,
               prop_loss_scale=1., marker_loss_scale=0., entropy_loss_scale=0.,
              ):
    super().__init__(dim, dim_latent, head_name)

    self.dim_counts = dim_counts
    self.temperature = temperature
    self.eps = eps
    self.prop_loss_scale = prop_loss_scale
    self.marker_loss_scale = marker_loss_scale
    self.entropy_loss_scale = entropy_loss_scale
    self.marker_groups = marker_groups

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
    logits = mixture_params.pop('logits', None)
    if logits is None:
      logits = tf.keras.initializers.RandomNormal()(shape=(self.dim, self.dim_counts))

    self.logits = tf.keras.Variable(logits,
                                    trainable=True,
                                    dtype=tf.float32,
                                    name=f'{head_name}_logits')

    log_total_count = mixture_params.pop('log_total_count', None)
    if log_total_count is None:
      log_total_count = tf.keras.initializers.Zeros()(shape=(1, self.dim_counts))

    self.log_total_count = tf.keras.Variable(log_total_count,
                                              trainable=True,
                                              dtype=tf.float32,
                                              name=f'{head_name}_counts' )

    inflated_loc_logits = mixture_params.pop('inflated_loc_logits', None)
    if inflated_loc_logits is None:
      inflated_loc_logits = tf.keras.initializers.Zeros()(shape=(1, dim_counts))

    self.inflated_loc_logits = tf.keras.Variable(inflated_loc_logits,
                                                 trainable=True,
                                                 dtype=tf.float32,
                                                 name=f'{head_name}_inflated')
    # <<< initialise mixtures <<<

    # get training variables
    self.t_vars = [*self.layers['mixture_logits'].trainable_variables,
                   *self.layers['encoder_classifier'].trainable_variables,
                   self.logits,
                   self.log_total_count,
                   self.inflated_loc_logits]


  def get_assignment_distribution(self, logits):
    return tfp.distributions.RelaxedOneHotCategorical(logits=logits, temperature=self.temperature)


  def get_mixture_distributions(self):
    _mix_shape = (self.dim, self.dim_counts)
    
    inflated_loc_probs = tf.math.sigmoid(self.inflated_loc_logits) * .75
    total_count = tf.math.exp(self.log_total_count) + self.eps

    return tfp.distributions.ZeroInflatedNegativeBinomial(
        logits=self.logits,
        total_count=tf.broadcast_to(total_count, _mix_shape),
        inflated_loc_probs=tf.broadcast_to(inflated_loc_probs, _mix_shape),
        require_integer_total_count=False,
    )


  #def get_mixture_mean(self):
  #  return tf.math.exp(self.log_total_count) * tf.math.exp(-self.logits)


  def decode_log_mixture_probs(self, latent):
    log_mixture_probs = self.layers['mixture_logits'](latent)

    # minimum topic proportion is EPS
    log_eps = tf.constant(tf.math.log(self.eps), shape=log_mixture_probs.shape)
    return tf.reduce_logsumexp(tf.stack([log_mixture_probs, log_eps]), axis=0)


  def decode(self, latent, data, sample=False):
    log_mixture_probs = self.decode_log_mixture_probs(latent)
    log_mixture_probs = tf.expand_dims(log_mixture_probs, 1) #(n_batch, 1, n_mix)

    mixtures = self.get_mixture_distributions()

    _broad_data = tf.expand_dims(data, -2) #(n_batch, n_subbatch, 1, n_counts)
    log_likelihoods = mixtures.log_prob(_broad_data) 
    log_likelihoods = tf.reduce_sum(log_likelihoods, axis=-1)

    assignment_sample = self.get_assignment_distribution(log_mixture_probs + log_likelihoods).sample() if sample else None

    return assignment_sample, log_likelihoods, log_mixture_probs


  def loss(self,
           data,
           latent,
           encoder_assignment_sample,
           encoder_assignment_logits,
           beta=1):
    _batch_size = data.shape[0]
             
    _, log_likelihoods, log_mixture_probs = self.decode(latent, data)

    kl_loss = tf.reduce_sum(
          tfp.distributions.Categorical(logits=encoder_assignment_logits).kl_divergence(
              tfp.distributions.Categorical(logits=log_mixture_probs)
              )
    )
    kl_loss /= _batch_size

    ll_loss = -tf.reduce_sum(encoder_assignment_sample * log_likelihoods)
    ll_loss /= _batch_size

    marker_loss = tf.constant(0.)
    if self.marker_groups is not None:
      probs = tf.math.softmax(encoder_assignment_logits, axis=-1)
      
      data = tf.expand_dims(data, axis=-2)
      markers = [tf.reduce_sum(tf.gather(data, marker_inds, axis=-1), axis=-1) for marker_inds, _ in self.marker_groups]
      antagonists = [tf.reduce_sum(tf.gather(data, antagonist_inds, axis=-1), axis=-1) for _, antagonist_inds in self.marker_groups]

      marker_loss = tf.reduce_mean(tf.stack(markers) * tf.stack(antagonists) * tf.expand_dims(probs, axis=0))

    entropy_loss = tf.constant(0.)
    if self.entropy_loss_scale != 0:
      entropy = tf.math.softmax(encoder_assignment_logits, axis=-1)
      entropy = tf.reduce_mean(entropy, axis=(0,1))
      entropy = entropy * tf.math.log(entropy)
      entropy_loss = -tf.reduce_sum(entropy)

    return tf.reduce_sum([self.prop_loss_scale*kl_loss,
                          ll_loss,
                          marker_loss * self.marker_loss_scale,
                          entropy_loss * self.entropy_loss_scale,
                          *self.layers['mixture_logits'].losses,
                          *self.layers['encoder_classifier'].losses])


  def encode(self, data):
    assignment_logits = self.layers['encoder_classifier'](data)
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
        'head_name':self.head_name,
        'head_type':self.head_type,
        'marker_groups':self.marker_groups,
        'temperature':self.temperature,
        'prop_loss_scale':self.prop_loss_scale,
        'marker_loss_scale':self.marker_loss_scale,
        'entropy_loss_scale':self.entropy_loss_scale,
        "layer_configs": {key: layer.get_config() for key, layer in self.layers.items()},
        'mixture_params':{
            'logits':self.logits.numpy().tolist(),
            'log_total_count':self.log_total_count.numpy().tolist(),
            'inflated_loc_logits':self.inflated_loc_logits.numpy().tolist(),
        },
    }
    return config

  def from_config(config):
    return FACTMx_head_ZINB_mixture3(**config)
