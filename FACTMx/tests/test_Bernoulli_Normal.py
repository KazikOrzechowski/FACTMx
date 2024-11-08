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

  dataset = tf.data.Dataset.zip( *[tf.data.Dataset.from_tensor_slices(view) for view in data] )

  val_data = []

  heads_config = [{'head_type':'Bernoulli', 'dim':1, 'head_name':'DNA'}] * n
  heads_config.append({'head_type':'MultiNormal', 'dim':3, 'head_name':'RNA'})

  model = FACTMx_model(dim_latent=2, heads_config=heads_config, beta=beta)

  return model, dataset, val_data
