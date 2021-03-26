# autograd_minimize

Wrapper around the minimize routine of scipy which uses the autograd capacities of 
tensorflow or pytorch to compute automatically the gradients, 
hessian vector products and hessians.

It also accepts functions of more than one variables as input.

## Installation 

`pip install autograd_minimize`

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
