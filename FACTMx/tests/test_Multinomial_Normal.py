def test_Multinomial_Normal(dim_latent, beta=1):
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

  model = FACTMx_model(dim_latent=2, heads_config=heads_config, beta=beta)

  return model, dataset, val_data, {'logits': logits}
