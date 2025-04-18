import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.keras.compat import keras


def wrap_model(model, pruning_params):
  model.layers = []
  pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(**pruning_params)
  
  #wrap encoder
  #for key, layer in model.encoder.layers.items():
  #  model.encoder.layers[key] = tfmot.sparsity.keras.prune_low_magnitude(layer, pruning_schedule)
  #  model.layers.append(model.encoder.layers[key])

  #wrap heads
  for i, head in enumerate(model.heads):
    for key, layer in head.layers.items():
      model.heads[i].layers[key] = tfmot.sparsity.keras.prune_low_magnitude(layer, pruning_schedule)
      model.layers.append(model.heads[i].layers[key])


def unwrap_model(model):
  model.layers = None

  #unwrap encoder
  #for key, layer in model.encoder.layers.items():
  #  model.encoder.layers[key] = tfmot.sparsity.keras.strip_pruning(layer)

  #wrap heads
  for i, head in enumerate(model.heads):
    for key, layer in head.layers.items():
      model.heads[i].layers[key] = tfmot.sparsity.keras.strip_pruning(layer)
  
