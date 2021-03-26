from time import time

import numpy as np
import tensorflow as tf
import torch
from autograd_minimize import minimize
from numpy.random import random
from numpy.testing import assert_almost_equal


def rosen_tst(backend='torch'):

    def rosen_tf(x):
        return tf.reduce_sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

    def rosen_torch(x):
        return (100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0).sum()
    
    func = rosen_tf if backend=='tf' else rosen_torch
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

    for method in [
        'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 
        'L-BFGS-B', 
        # 'TNC', # Not precise enough
        # 'COBYLA', # Not precise enough
        'SLSQP', 'trust-constr',
        # 'dogleg',  # requires positive semi definite hessian, not the case here the problem is not convex
        'trust-ncg', 
        'trust-exact', # requires hessian
        'trust-krylov'
        ]:

        tic = time()    
        res = minimize(func, x0, 
            backend=backend,
            precision='float64',
            method=method,
            tol=1e-8)

        print(method, time()-tic, np.mean(res.x-1))
        assert_almost_equal(res.x, 1, decimal=5)

def test_rosen_tf():
    rosen_tst('tf')

def test_rosen_torch():
    rosen_tst('torch')  

def test_cstr_opt():
    fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2
    cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},
            {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
            {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})
    bnds = ((0, None), (0, None))

    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    res = minimize(fun, np.array([2, 0]), method='SLSQP', bounds=bnds,
               constraints=cons)

    assert_almost_equal(res.x, np.array([1.4, 1.7]), decimal=6)


def test_matrix_decomposition(shape=(10,20), inner_shape=3, method=None):
    U = random((shape[0], inner_shape))
    V = random((inner_shape, shape[1]))
    prod = U@V

    def model(U=None, V=None):
        return tf.reduce_mean((U@V-tf.constant(prod, dtype=tf.float32))**2)

    def model_torch(smv=None, smp=None):
        return ((smv[:, None, :, None]*smp[None, :, None, :]-torch.tensor(Z, dtype=torch.float32))**2).mean()

    x0 = {'U': random((shape[0], inner_shape)), 'V': random((inner_shape, shape[1]))}

    tic = time()
    res = minimize(model, x0, method=method)
    print(method, time()-tic, res.fun)

    x0 = [random((shape[0], inner_shape)), random((inner_shape, shape[1]))]

    tic = time()
    res = minimize(model, x0, method=method)
    print(method, time()-tic, res.fun)


