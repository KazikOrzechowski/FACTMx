def test_Normal_2D(sd1, sd2, covar=0., beta=1):
  data_loc = tf.zeros((2,))
  data_scale = tfp.math.fill_triangular([sd2, covar, sd1])

  data = [tfp.distributions.MultivariateNormalTriL(data_loc, data_scale).sample((10000,))]
  dataset = tf.data.Dataset.from_tensor_slices(data)

  val_data = [tfp.distributions.MultivariateNormalTriL(data_loc, data_scale).sample((1000,))]

  heads_config = [{'head_type':'MultiNormal', 'dim':2, 'head_name':'RNA'}]

  model = FACTMx_model(dim_latent=1, heads_config=heads_config, beta=beta)

  return model, dataset, val_data
