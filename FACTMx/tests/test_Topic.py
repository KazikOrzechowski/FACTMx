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

  data = [tfp.distributions.Multinomial(trials, logits=obs_logits).sample()]
  dataset = tf.data.Dataset.from_tensor_slices(data)

  val_data = []

  heads_config = [{'head_type':'Topic', 'dim':4,
                   'dim_words':3, 'head_name':'IF'}]

  model = FACTMx_model(dim_latent=dim_latent, heads_config=heads_config, beta=beta)

  return model, dataset, val_data, {'topic_log_densities':topic_log_densities,
                                    'assignments':assignments}
