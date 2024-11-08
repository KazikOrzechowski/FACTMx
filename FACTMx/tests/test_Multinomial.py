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

  model = FACTMx_model(dim_latent, heads_config=heads_config, beta=beta)

  return model, dataset, val_data
