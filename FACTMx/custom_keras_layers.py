"""
Custom Keras layers used by FACTMx model components.

The layers in this module are intentionally small wrappers around Keras
functionality.  They exist so model heads and encoders can request special
feature transformations or trainable constant responses while still being
serializable through Keras configuration dictionaries.
"""

import tensorflow as tf
import tensorflow.keras as keras

@keras.utils.register_keras_serializable()
class ConstantResponse(keras.layers.Layer):
  """
  A trainable layer that ignores its input and returns a learned vector.
  
  This is useful for scale or dispersion parameters that should be shared across
  all samples rather than conditioned on the current input batch.
  """
  def __init__(self,
               units,
               input_dim=None,
               bias_initializer='zeros',
               trainable=True,
               activation='linear',
               **kwargs):
    """
    Create a constant-response layer.
    
    Args:
      units: Number of output units in the learned response vector.
      input_dim: Kept for API compatibility; the input is not used.
      bias_initializer: Keras initializer for the learned response vector.
      trainable: Whether the response vector should be optimized.
      activation: Activation applied to the response vector before returning it.
      **kwargs: Additional Keras layer keyword arguments.
    """
    super().__init__()
    self.b = self.add_weight(shape=(units,),
                             initializer=keras.initializers.deserialize(bias_initializer),
                             trainable=trainable)
    self.activation = keras.activations.get(activation)

  def call(self, inputs):
    """Return the learned response vector with the configured activation."""
    return self.activation(self.b)

  def get_prunable_weights(self):
    """Return an empty pruning list because this layer has no kernel weights."""
    return []



@keras.utils.register_keras_serializable()
class QuadraticFeatures(keras.layers.Layer):
  """
  Expand a feature vector into all pairwise quadratic products.
  
  For an input with ``dim_features`` entries, the output contains
  ``dim_features ** 2`` flattened dot-product features.  Batch dimensions are
  preserved.
  """
  def __init__(self,
               **kwargs):
    """Initialise the internal dot-product layer used for quadratic expansion."""
    super().__init__()
    self.Dot = keras.layers.Dot(axes=-1)

  def call(self, inputs):
    """Compute flattened pairwise products for the final feature dimension."""
    # Keep any leading batch dimensions, but flatten them temporarily when the
    # incoming tensor has more than one batch-like axis.
    *batch, dim_features = list(inputs.shape)
    _output_shape = [*batch, -1]

    if len(inputs.shape) > 2:
      inputs = tf.reshape(inputs, (-1, dim_features))
    inputs = tf.expand_dims(inputs, -1)
    
    outputs = self.Dot([inputs, inputs])
    outputs = tf.reshape(outputs, _output_shape)
    return outputs

  def compute_output_shape(self, input_shape):
      """Return the expected output shape after quadratic feature expansion."""
      *batch, dim_features = input_shape
      return (*batch, dim_features ** 2)

  def build(self, input_shape):
      """Delegate build-time setup to the Keras base layer."""
      super(QuadraticFeatures, self).build(input_shape)

  def get_prunable_weights(self):
    """Return an empty pruning list because the layer has no prunable weights."""
    return []
