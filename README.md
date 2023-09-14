# autograd-minimize

autograd-minimize is a wrapper around the minimize routine of scipy which uses the autograd capacities of 
jax, tensorflow or pytorch to compute automatically the gradients, 
hessian vector products and hessians.

It also accepts functions of more than one variables as input.

## Installation 

`pip install autograd-minimize`

## Basic usage

It uses tensorflow as the default backend:

```
import tensorflow as tf
from autograd_minimize import minimize

def rosen_tf(x):
    return tf.reduce_sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

res = minimize(rosen_tf, np.array([0.,0.]))
print(res.x)
>>> array([0.99999912, 0.99999824])
```

But you can also use pytorch: 

```
import torch
from autograd_minimize import minimize
import numpy as np

def rosen_torch(x):
    return (100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0).sum()
    
res = minimize(rosen_torch, np.array([0.,0.]), backend='torch')
print(res.x)
>>> array([0.99999912, 0.99999824])
```

Or jax:
```
import numpy as np
from autograd_minimize import minimize

rosen_jax=lambda x: (100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0).sum()
res = minimize(rosen_jax, np.array([0.,0.]), backend='jax')
print(res.x)
>>> array([0.99999912, 0.99999824])
```

You can also try other optimization methods such as Newton-CG which uses 
automatic computation of the hessian vector product (hvp). Let's as well 
increase the precision of hvp and gradient computation to float64 and the tolerance to 1e-8: 

```
import numpy as np
res = minimize(rosen_tf, np.array([0.,0.]), method='Newton-CG', precision='float64', tol=1e-8)
print(np.mean(res.x-1))
>>> -2.6886433635020524e-09
```

Or we can use the trust-exact method (with automatic computation of the hessian): 

```
import numpy as np
res = minimize(rosen_tf, np.array([0.,0.]), method='trust-exact', precision='float64', tol=1e-8)
print(np.mean(res.x-1))
>>> -1.6946999359390702e-12
```

Let's now try to do matrix factorization. In this case it is much easier to deal with a function with two inputs, where the input should be a dict or a list with a similar signature as the function: 

```
shape = (10, 15)
inner_shape=3
from numpy.random import random
U = random((shape[0], inner_shape))
V = random((inner_shape, shape[1]))
prod = U@V

def mat_fac(U, V):
    return tf.reduce_mean((U@V-tf.constant(prod, dtype=tf.float32))**2)

x0 = {'U': -random((shape[0], inner_shape)), 'V': random((inner_shape, shape[1]))}

tic = time()
res = minimize(mat_fac, x0)
print(res.fun)
>>> 6.136937713563384e-08

```

## Bounds

You can also set bounds (only for the methods: L-BFGS-B, TNC, SLSQP, Powell, and trust-constr):

If bounds is a tuple, the same bound is applied to all variables:

```
res = minimize(mat_fac, x0, bounds=(None, 0))
print(res.x['U'].mean())
>>> -0.6171053993128699
```

You can apply bounds only to a subset of variables by using a list or a dict (but it should be the same as the format of input x0):

```
res = minimize(mat_fac, x0, bounds={'U': (None, 0), 'V': (-1, None)})
print(res.x['U'].mean(), res.x['V'].mean())
>>> -0.8173837691822693 0.11222992115637932
```

Inside each variable of the dict/list, you can pass a numpy array or a list of 
bounds which the same shape or len as the variable to specify in more details the bounds:

```
res = minimize(mat_fac, x0, bounds={'U': (0, None), 'V': [(0, None)]*inner_shape*shape[1]})
```

## Keras models

You can also optimize keras models by transforming them into a function of their parameters, using 
`autograd_minimize.tf_wrapper.tf_function_factory`:

```
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
func, params = tf_function_factory(model, tf.keras.losses.MSE, X, y)

# Minimization
res = minimize(func, params, method='L-BFGS-B')
```
Note that you can do the same on torch models by replacing `autograd_minimize.tf_wrapper.tf_function_factory` by `autograd_minimize.torch_wrapper.torch_function_factory`.

## Constraints

And you can set constraints (with automatic computation of the jacobian). An example is given in `examples/multiknapsack`, where the (relaxed) multiknapsack problem is solved.

## ToDo

* Adds comparison with LBFGS from pytorch or keras