import numpy as np
import tensorflow as tf
from tensorflow import keras

from autograd_minimize import minimize
from autograd_minimize.tf_wrapper import tf_function_factory

# Prepares data
X = np.random.random((200, 2))
y = X[:, :1] * 2 + X[:, 1:] * 0.4 - 1

# Creates model
model = keras.Sequential([keras.Input(shape=2), keras.layers.Dense(1)])

# Transforms model into a function of its parameter
func, params, names = tf_function_factory(model, tf.keras.losses.MSE, X, y)

# Minimization
res = minimize(func, params, method="trust-constr")

print("Fitted parameters:")
print([var.numpy() for var in model.trainable_variables])

print(f"mae: {tf.reduce_mean(tf.abs(model(X)-y))}")
