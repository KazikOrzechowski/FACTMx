import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.keras.compat import keras


def wrap_model(model, pruning_params):
  model.prunable_layers = []
  pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(**pruning_params)
  
  #wrap encoder
  for key, layer in self.encoder.layers.items():
    self.encoder.layers[key] = tfmot.sparsity.keras.prune_low_magnitude(layer, pruning_schedule)
    model.prunable_layers.append(self.encoder.layers[key])

  #wrap heads
  for i, head in enumerate(self.heads):
    for key, layer in head.layers.items():
      self.heads[i].layers[key] = tfmot.sparsity.keras.prune_low_magnitude(layer, pruning_schedule)
      model.prunable_layers.append(self.heads[i].layers[key])


def unwrap_model(model):
  model.prunable_layers = None

  #unwrap encoder
  for key, layer in self.encoder.layers.items():
    self.encoder.layers[key] = tfmot.sparsity.keras.strip_pruning(layer)

  #wrap heads
  for i, head in enumerate(self.heads):
    for key, layer in head.layers.items():
      self.heads[i].layers[key] = tfmot.sparsity.keras.strip_pruning(layer)
  
