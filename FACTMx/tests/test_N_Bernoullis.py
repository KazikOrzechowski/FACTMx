def test_N_Bernoullis(n, scale=1, beta=1):
  latent_loc = tf.zeros((1,))
  latent_scale = tf.eye(1)

  latent = tfp.distributions.MultivariateNormalTriL(latent_loc, latent_scale).sample((10000,))

  logits = scale * latent

  data = [tfp.distributions.Bernoulli(logits=logits).sample() for _ in range(n)]
  data = [tf.cast(view, dtype='float32') for view in data]

  dataset = tf.data.Dataset.from_tensor_slices(data)

  val_data = []

  heads_config = [{'head_type':'Bernoulli', 'dim':1, 'head_name':'DNA'}] * n

  model = FACTMx_model(dim_latent=1, heads_config=heads_config, beta=beta)

  return model, dataset, val_data
