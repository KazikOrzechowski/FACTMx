def test_Normal_ND_with_noise(sd, n_dim, sd_noise=0.1, noise_dim=10, dim_latent=1, beta=1):
  data_loc = tf.zeros((n_dim + noise_dim,))
  data_scale = tf.linalg.diag([sd]*n_dim + [sd_noise]*noise_dim)

  data = [tfp.distributions.MultivariateNormalTriL(data_loc, data_scale).sample((10000,))]
  dataset = tf.data.Dataset.from_tensor_slices(data)

  val_data = [tfp.distributions.MultivariateNormalTriL(data_loc, data_scale).sample((1000,))]

  heads_config = [{'head_type':'MultiNormal', 'dim':n_dim+noise_dim, 'head_name':'RNA'}]

  model = FACTMx_model(dim_latent=dim_latent, heads_config=heads_config, beta=beta)

  return model, dataset, val_data
