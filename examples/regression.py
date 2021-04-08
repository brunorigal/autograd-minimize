import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from autograd_minimize.tf_wrapper import tf_function_factory
from autograd_minimize import minimize 
import tensorflow as tf

#### Prepares data
X = np.random.random((200, 2))
y = X[:,:1]*2+X[:,1:]*0.4-1

#### Creates model
model = keras.Sequential([keras.Input(shape=2),
                          layers.Dense(1)])

# Transforms model into a function of its parameter
func = tf_function_factory(model, tf.keras.losses.MSE, X, y)

# Minimization
res = minimize(func, [var.numpy() for var in model.trainable_variables], method='L-BFGS-B')

# Assigns minimized params
for i, param in enumerate(res.x):
    model.trainable_variables[i].assign(param)

print('Fitted parameters:')
print([var.numpy() for var in model.trainable_variables])
