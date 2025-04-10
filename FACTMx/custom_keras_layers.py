try:
  from tensorflow_model_optimization.python.core.keras.compat import keras
except ImportError:
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
