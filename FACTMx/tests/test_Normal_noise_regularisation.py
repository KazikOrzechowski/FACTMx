def test_Normal_noise_regularisation(sd, n_dim,
                                     sd_noise=0.1, noise_dim=10,
                                     dim_latent=1, beta=1,
                                     l1=0.1, l2=0.1):
  #create data: Normally distributed, n_dim informative dimensions with sd, noise_dim uninformative dimensions
  data_loc = tf.zeros((n_dim + noise_dim,))
  data_scale = tf.linalg.diag([sd]*n_dim + [sd_noise]*noise_dim)

  data = [tfp.distributions.MultivariateNormalTriL(data_loc, data_scale).sample((10000,))]
  dataset = tf.data.Dataset.from_tensor_slices(data)

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
  model = FACTMx_model(dim_latent=dim_latent,
                       encoder_config=encoder_config,
                       heads_config=heads_config,
                       beta=beta)

  return model, dataset, val_data
