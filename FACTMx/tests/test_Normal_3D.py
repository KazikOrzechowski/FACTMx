def test_Normal_3D(sd1, sd2, covar=0., dim_latent=1, beta=1):
  data_loc = tf.zeros((3,))
  data_scale = tfp.math.fill_triangular([1., 0, 0,
                                         sd1,
                                         sd2, covar])

  data = [tfp.distributions.MultivariateNormalTriL(data_loc, data_scale).sample((10000,))]
  dataset = tf.data.Dataset.from_tensor_slices(data)

  val_data = [tfp.distributions.MultivariateNormalTriL(data_loc, data_scale).sample((1000,))]

  heads_config = [{'head_type':'MultiNormal', 'dim':3, 'head_name':'RNA'}]

  model = FACTMx_model(dim_latent=dim_latent, heads_config=heads_config, beta=beta)

  return model, dataset, val_data
