import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from FACTMx.custom_keras_layers import ConstantResponse

def test_Normal_2D(sd1, sd2, covar=0., beta=1):
  data_loc = tf.zeros((2,))
  data_scale = tfp.math.fill_triangular([sd2, covar, sd1])

  data_normal = tfp.distributions.MultivariateNormalTriL(data_loc, data_scale).sample((10000,))
  dataset = tf.data.Dataset.from_tensor_slices((data_normal,))

  val_data = [tfp.distributions.MultivariateNormalTriL(data_loc, data_scale).sample((1000,))]

  heads_config = [{'head_type':'MultiNormal', 'dim':2, 'head_name':'RNA'}]

  model_config = {
    'dim_latent': 1,
    'heads_config': heads_config,
    'beta': beta
  }
  
  return model_config, dataset, val_data, {}


def test_Normal_2D_bias(sd1, sd2, covar=0., beta=1):
  data_loc = tf.ones((2,))
  data_scale = tfp.math.fill_triangular([sd2, covar, sd1])

  data_normal = tfp.distributions.MultivariateNormalTriL(data_loc, data_scale).sample((10000,))
  dataset = tf.data.Dataset.from_tensor_slices((data_normal,))

  val_data = [tfp.distributions.MultivariateNormalTriL(data_loc, data_scale).sample((1000,))]

  heads_config = [{'head_type':'MultiNormal', 'dim':2, 'head_name':'RNA'}]

  model_config = {
    'dim_latent': 1, 
    'heads_config': heads_config,
    'beta': beta
  }

  return model_config, dataset, val_data, {}


def test_Normal_3D(sd1, sd2, covar=0., dim_latent=1, beta=1):
  data_loc = tf.zeros((3,))
  data_scale = tfp.math.fill_triangular([1., 0, 0,
                                         sd1,
                                         sd2, covar])

  data_normal = tfp.distributions.MultivariateNormalTriL(data_loc, data_scale).sample((10000,))
  dataset = tf.data.Dataset.from_tensor_slices((data_normal,))

  val_data = [tfp.distributions.MultivariateNormalTriL(data_loc, data_scale).sample((1000,))]

  heads_config = [{'head_type':'MultiNormal', 'dim':3, 'head_name':'RNA'}]

  model_config = {
    'dim_latent': dim_latent, 
    'heads_config': heads_config, 
    'beta': beta
  }

  return model_config, dataset, val_data, {}


def test_Normal_ND_with_noise(sd, n_dim, sd_noise=0.1, noise_dim=10, dim_latent=1, beta=1):
  data_loc = tf.zeros((n_dim + noise_dim,))
  data_scale = tf.linalg.diag([sd]*n_dim + [sd_noise]*noise_dim)

  data_normal = tfp.distributions.MultivariateNormalTriL(data_loc, data_scale).sample((10000,))
  dataset = tf.data.Dataset.from_tensor_slices((data_normal,))

  val_data = [tfp.distributions.MultivariateNormalTriL(data_loc, data_scale).sample((1000,))]

  heads_config = [{'head_type':'MultiNormal', 'dim':n_dim+noise_dim, 'head_name':'RNA'}]

  model_config = {
    'dim_latent': dim_latent, 
    'heads_config': heads_config, 
    'beta': beta
  }

  return model_config, dataset, val_data, {}


def test_N_Bernoullis(n, scale=1, beta=1):
  latent_loc = tf.zeros((1,))
  latent_scale = tf.eye(1)

  latent = tfp.distributions.MultivariateNormalTriL(latent_loc, latent_scale).sample((10000,))

  logits = scale * latent

  data_bernoulli = [tfp.distributions.Bernoulli(logits=logits).sample() for _ in range(n)]
  data_bernoulli = [tf.cast(view, dtype='float32') for view in data]

  dataset = tf.data.Dataset.from_tensor_slices(tuple(data_bernoulli))

  val_data = []

  heads_config = [{'head_type':'Bernoulli', 'dim':1, 'head_name':'DNA'}] * n

  model_config = {
    'dim_latent': 1, 
    'heads_config': heads_config, 
    'beta': beta
  }

  return model_config, dataset, val_data, {'logits': logits}


def test_Bernoulli_Normal(n, beta=1):
  latent_loc = tf.zeros((2,))
  latent_scale = tf.eye(2)

  latent = tfp.distributions.MultivariateNormalTriL(latent_loc, latent_scale).sample((10000,))

  #Bernoulli
  scale = np.array([[1.], [-1]])

  logits = latent @ scale

  data = [tfp.distributions.Bernoulli(logits=logits).sample() for _ in range(n)]
  data = [tf.cast(view, dtype='float32') for view in data]

  #Normal
  decode_mat_loc = tf.constant([[3., 0., 0.],
                                [0., 2., 1.]])

  data_scale = tf.constant([[0.1, 0., 0.],
                            [0., 0.1, 0.],
                            [0., 0., 0.1]])

  data_loc = latent @ decode_mat_loc
  data.append( tfp.distributions.MultivariateNormalTriL(data_loc, data_scale).sample() )

  dataset = tf.data.Dataset.from_tensor_slices(tuple(data))

  val_data = []

  heads_config = [{'head_type':'Bernoulli', 'dim':1, 'head_name':'DNA'}] * n
  heads_config.append({'head_type':'MultiNormal', 'dim':3, 'head_name':'RNA'})

  model_config = {
    'dim_latent': 2, 
    'heads_config': heads_config, 
    'beta': beta
  }

  return model_config, dataset, val_data, {'logits': logits}


def test_combined(dim_latent, beta=1):
  latent_loc = tf.zeros((2,))
  latent_scale = tf.eye(2)

  latent = tfp.distributions.MultivariateNormalTriL(latent_loc, latent_scale).sample((1000,))

  #Bernoulli
  scale = np.array([[1, 0],
                    [0, -1.]])

  logits = latent @ scale

  multi_counts = tfp.distributions.Poisson(6).sample(1000)
  multi_obs = tfp.distributions.Multinomial(total_count=multi_counts, logits=logits).sample()
  multi_obs = tf.cast(multi_obs, dtype='float32')

  #Normal
  decode_mat_loc = tf.constant([[3., 0., 0.],
                                [0., 2., 1.]])

  data_scale = tf.constant([[0.1, 0., 0.],
                            [0., 0.1, 0.],
                            [0., 0., 0.1]])

  data_loc = latent @ decode_mat_loc
  normal_obs = tfp.distributions.MultivariateNormalTriL(data_loc, data_scale).sample()

  dataset = tf.data.Dataset.from_tensor_slices( ((multi_obs, multi_counts), normal_obs) )

  val_data = []

  heads_config = [{'head_type':'Multinomial', 'dim':1, 'head_name':'DNA'}]
  heads_config.append({'head_type':'MultiNormal', 'dim':3, 'head_name':'RNA'})

  model_config = {
    'dim_latent': 2, 
    'heads_config': heads_config,
    'beta': beta
  }

  return model_config, dataset, val_data, {'logits': logits}


def test_Multinomial(dim_latent, beta=1):
  latent_loc = tf.zeros((2,))
  latent_scale = tf.eye(2)

  latent = tfp.distributions.MultivariateNormalTriL(latent_loc, latent_scale).sample((1000,))

  decode_mat_loc = tf.constant([[1., 1, 0],
                                [0., 0, 1]])

  data_scale = tf.constant([[0.1, 0., 0.],
                            [0., 0.1, 0.],
                            [0., 0., .1]])

  data_loc = latent @ decode_mat_loc

  #draw logits
  logits = tfp.distributions.MultivariateNormalTriL(data_loc, data_scale).sample()

  padded_logits = tf.pad(logits, tf.constant([[0, 0], [1, 0]]), 'CONSTANT')

  #observations
  counts = tfp.distributions.Poisson(6).sample((1000,))

  observations = tfp.distributions.Multinomial(total_count=counts, logits=padded_logits).sample()

  dataset = tf.data.Dataset.from_tensor_slices(((observations, counts),))

  val_data = []

  heads_config = [{'head_type':'Multinomial',
                   'dim':3,
                   'head_name':'rna'}]

  model_config = {
    'dim_latent': dim_latent,
    'heads_config': heads_config,
    'beta': beta
  }

  return model_config, dataset, val_data, {'logits': padded_logits}


def test_Topic(dim_latent, beta=1, n_patients=100, n_obs=1000):
  latent_loc = tf.zeros((2,))
  latent_scale = tf.eye(2)

  latent = tfp.distributions.MultivariateNormalTriL(latent_loc, latent_scale).sample((n_patients,))

  #CTM
  decode_mat_loc = tf.constant([[1., 0., 0.],
                                [0., .5, 0.]])

  data_scale = tf.constant([[0.1, 0., 0.],
                            [0., 0, 0.],
                            [0., 0.1, 0]])

  data_loc = latent @ decode_mat_loc

  #draw relative densities
  topic_log_densities = tfp.distributions.MultivariateNormalTriL(data_loc, data_scale).sample()

  paddings_proportions = tf.constant([[0, 0], [1, 0]])
  topic_log_densities = tf.pad(topic_log_densities, paddings_proportions, 'CONSTANT')

  #draw assignments
  assignments = tfp.distributions.RelaxedOneHotCategorical(temperature=.0001, logits=topic_log_densities).sample(n_obs)
  assignments = tf.transpose(assignments, [1, 0, 2])

  #make topic profiles
  topic_profiles = tf.constant([[-2, 2, 0., 0.],
                                [-2, 0., 2, 0.]])

  paddings_profiles = tf.constant([[1, 0], [0, 0]])
  padded_profiles = tf.pad(topic_profiles, paddings_profiles, 'CONSTANT')

  #draw observations
  obs_logits = assignments @ tf.transpose(padded_profiles)

  trials = tfp.distributions.Poisson(6).sample((n_patients,n_obs))

  data = tfp.distributions.Multinomial(trials, logits=obs_logits).sample()
  dataset = tf.data.Dataset.from_tensor_slices((data,))

  val_data = []

  heads_config = [{'head_type':'TopicModel', 'dim':3,
                   'dim_words':3, 'head_name':'IF'}]

  model_config = {
    'dim_latent': dim_latent, 
    'heads_config': heads_config, 
    'beta': beta
  }

  return model_config, dataset, val_data, {'topic_log_densities':topic_log_densities,
                                           'assignments':assignments}


def test_Normal_noise_regularisation(sd, n_dim,
                                     sd_noise=0.1, noise_dim=10,
                                     dim_latent=1, beta=1,
                                     l1=0.1, l2=0.1):
  #create data: Normally distributed, n_dim informative dimensions with sd, noise_dim uninformative dimensions
  data_loc = tf.zeros((n_dim + noise_dim,))
  data_scale = tf.linalg.diag([sd]*n_dim + [sd_noise]*noise_dim)

  data = tfp.distributions.MultivariateNormalTriL(data_loc, data_scale).sample((10000,))
  dataset = tf.data.Dataset.from_tensor_slices((data,))

  val_data = [tfp.distributions.MultivariateNormalTriL(data_loc, data_scale).sample((1000,))]

  #setup regularization
  reg = tf.keras.regularizers.L1L2(l1=l1, l2=l2)

  #setup Sequential model configurations for
  #encoding
  encode_model_loc = tf.keras.Sequential(
                              [tf.keras.Input(shape=(n_dim + noise_dim,)),
                               tf.keras.layers.Dense(units=dim_latent,
                                                     kernel_initializer='orthogonal',
                                                     kernel_regularizer=reg)]
      )
  encode_model_scale = tf.keras.Sequential(
                              [tf.keras.Input(shape=(n_dim + noise_dim,)),
                               ConstantResponse(units=dim_latent,
                                                activation='relu',
                                                bias_initializer='zeros')]
      )
  #decoding
  decode_model_loc = tf.keras.Sequential(
                              [tf.keras.Input(shape=(dim_latent,)),
                               tf.keras.layers.Dense(units=n_dim + noise_dim,
                                                     kernel_initializer='orthogonal',
                                                     kernel_regularizer=reg)]
      )
  decode_model_scale = tf.keras.Sequential(
                              [tf.keras.Input(shape=(dim_latent,)),
                               ConstantResponse(units=n_dim + noise_dim,
                                                activation='relu',
                                                bias_initializer='zeros')]
      )

  #create encoder configuration dictionary
  encoder_config = {'dim_latent':dim_latent,
                    'head_dims':(n_dim + noise_dim),
                    'layer_configs':{'loc':encode_model_loc.get_config(),
                                     'scale':encode_model_scale.get_config()}}

  #create a list of head configuration dictionaries
  heads_config = [{'head_type':'MultiNormal',
                   'dim':n_dim+noise_dim,
                   'head_name':'RNA',
                   'layer_configs':{'loc':decode_model_loc.get_config(),
                                     'scale':decode_model_scale.get_config()}
                   }]

  #create model with specified configuration of encoder and heads
  model_config = {
    'dim_latent': dim_latent,
    'encoder_config': encoder_config,
    'heads_config': heads_config,
    'beta': beta
  }

  return model_config, dataset, val_data, {}
