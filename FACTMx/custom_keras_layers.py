"""Custom Keras layers used by FACTMx model components."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

import tensorflow as tf
import tensorflow.keras as keras

TensorLike = Any


@keras.utils.register_keras_serializable()
class ConstantResponse(keras.layers.Layer):
  """Keras layer returning a learned constant response for every input row.

  Args:
    units: Number of output units.
    input_dim: Optional input feature count retained for config compatibility.
    bias_initializer: Keras initializer for the learned constant.
    trainable: Whether the constant parameter is trainable.
    activation: Activation applied to the learned constant.
    **kwargs: Additional Keras layer keyword arguments.
  """

  def __init__(
      self,
      units: int,
      input_dim: Optional[int] = None,
      bias_initializer: Any = 'zeros',
      trainable: bool = True,
      activation: str = 'linear',
      **kwargs: Any,
  ) -> None:
    super().__init__(**kwargs)
    self.units = units
    self.input_dim = input_dim
    self.bias_initializer = bias_initializer
    self.b = self.add_weight(
        shape=(units,),
        initializer=keras.initializers.deserialize(bias_initializer),
        trainable=trainable,
    )
    self.activation = keras.activations.get(activation)

  def call(self, inputs: TensorLike) -> tf.Tensor:
    """Return the activated constant vector."""
    return self.activation(self.b)

  def get_prunable_weights(self) -> list[tf.Variable]:
    """Return prunable weights for TensorFlow Model Optimization Toolkit."""
    return []


@keras.utils.register_keras_serializable()
class QuadraticFeatures(keras.layers.Layer):
  """Layer producing flattened pairwise products of input features."""

  def __init__(self, **kwargs: Any) -> None:
    super().__init__(**kwargs)
    self.Dot = keras.layers.Dot(axes=-1)

  def call(self, inputs: TensorLike) -> tf.Tensor:
    """Return flattened quadratic features for ``inputs``."""
    *batch, dim_features = list(inputs.shape)
    output_shape = [*batch, -1]

    if len(inputs.shape) > 2:
      inputs = tf.reshape(inputs, (-1, dim_features))
    inputs = tf.expand_dims(inputs, -1)

    outputs = self.Dot([inputs, inputs])
    outputs = tf.reshape(outputs, output_shape)
    return outputs

  def compute_output_shape(self, input_shape: Sequence[int]) -> Tuple[int, ...]:
    """Return the output shape for a given input shape."""
    *batch, dim_features = input_shape
    return (*batch, dim_features ** 2)

  def build(self, input_shape: Sequence[int]) -> None:
    """Build the layer."""
    super().build(input_shape)

  def get_prunable_weights(self) -> list[tf.Variable]:
    """Return prunable weights for TensorFlow Model Optimization Toolkit."""
    return []
