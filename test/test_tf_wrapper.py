import pdb
import tensorflow as tf
import numpy as np
from autograd_minimize import minimize
from time import time
from numpy.random import random
from numpy.testing import assert_almost_equal


def test_rosen():

    def rosen(x):
        return tf.reduce_sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    res = minimize(rosen, x0, 
    # method='Nelder-Mead', 
    method='BFGS',
    tol=1e-7)

    assert_almost_equal(res.x, 1, decimal=6)


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


def test_matrix_tensor_product():
    smp = random((5, 5))
    smv = random((3, 3))

    Z = smv[:, None, :, None]*smp[None, :, None, :]

    def model(smv=None, smp=None):
        return tf.reduce_mean((smv[:, None, :, None]*smp[None, :, None, :]-tf.constant(Z, dtype=tf.float32))**2)

    x0 = {'smv': random((3, 3)), 'smp': random((5, 5))}

    tic = time()
    res = minimize(model, x0)
    print(time()-tic)

    x0 = [random((3, 3)), random((5, 5))]

    tic = time()
    res = minimize(model, x0)
    print(time()-tic)

    
