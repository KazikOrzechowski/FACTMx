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
    *_preshape, _last = inputs.shape
    
    inputs = tf.expand_dims(inputs, -1)
    
    outputs = self.Dot([inputs, inputs])
    outputs = tf.reshape(outputs, _preshape + (_last ** 2,))
    return outputs

  def get_prunable_weights(self):
    return []
