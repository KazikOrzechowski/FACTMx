import tensorflow as tf
import tensorflow.keras as keras

@keras.utils.register_keras_serializable()
class ConstantResponse(keras.layers.Layer):
  def __init__(self,
               units,
               input_dim=None,
               bias_initializer='zeros',
               trainable=True,
               activation='linear',
               **kwargs):
    super().__init__()
    self.b = self.add_weight(shape=(units,),
                             initializer=bias_initializer,
                             trainable=trainable)
    self.activation = keras.activations.get(activation)

  def call(self, inputs):
    return self.activation(self.b)

  def get_prunable_weights(self):
    return []



@keras.utils.register_keras_serializable()
class QuadraticFeatures(keras.layers.Layer):
  def __init__(self,
               **kwargs):
    super().__init__()
    self.Dot = keras.layers.Dot(axes=-1)

  def call(self, inputs):
    *batch, dim_features = list(inputs.shape)
    _output_shape = [*batch, -1]

    if len(inputs_shape) > 2:
      inputs = tf.reshape(inputs, (-1, dim_features))
    inputs = tf.expand_dims(inputs, -1)
    
    outputs = self.Dot([inputs, inputs])
    outputs = tf.reshape(outputs, _output_shape)
    return outputs

  def compute_output_shape(self, input_shape):
      *batch, dim_features = input_shape
      return (*batch, dim_features ** 2)

  def get_prunable_weights(self):
    return []
