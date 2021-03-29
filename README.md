# autograd_minimize

Wrapper around the minimize routine of scipy which uses the autograd capacities of 
tensorflow or pytorch to compute automatically the gradients, 
hessian vector products and hessians.

It also accepts functions of more than one variables as input.

## Installation 

`pip install git+https://github.com/brunorigal/autograd_minimize.git`

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

def rosen_torch(x):return (100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0).sum()
    
res = minimize(rosen_torch, np.array([0.,0.]), backend='torch')
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

Let's now try to do matrix factorization. In this case it is much easier to deal with a function with two inputs, where the input should be a dict or a list with a similar singature as the function: 

```
shape = (10, 15)
inner_shape=3
from numpy.random import random
U = random((shape[0], inner_shape))
V = random((inner_shape, shape[1]))
prod = U@V

def mat_fac(U=None, V=None):
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

## Constraints

And you can set constraints (with automatic computation of the jacobian). An example is given in examples/multiknapsack, where the (relaed multiknapsack problem is solved).
