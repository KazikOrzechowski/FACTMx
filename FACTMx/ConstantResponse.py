import tensorflow as tf
import keras

@keras.saving.register_keras_serializable()
class ConstantResponse(tf.keras.layers.Layer):
  def __init__(self, 
               units, 
               input_dim=None, 
               bias_initializer='zeros', 
               trainable=True, 
               activation='linear',
               **kwargs):
    '''
    Layer for including a constant in a Sequential model
    '''
    super().__init__()
    self.b = self.add_weight(shape=(units,), 
                             initializer=bias_initializer, 
                             trainable=trainable)
    self.activation = tf.keras.activations.get(activation)

  def call(self, inputs):
    return self.activation(self.b)
