from time import time

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from autograd_minimize import minimize
from autograd_minimize.tf_wrapper import tf_function_factory
from autograd_minimize.torch_wrapper import torch_function_factory
from numpy.random import random
from numpy.testing import assert_almost_equal
from tensorflow import keras
from tensorflow.keras import layers


def rosen_tst(backend='torch'):
    """
    Adapated from: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """

    def rosen_tf(x):
        return tf.reduce_sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

    def rosen_torch(x):
        return (100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0).sum()

    func = rosen_tf if backend == 'tf' else rosen_torch
    x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

    for method in [
        'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG',
        'L-BFGS-B',
        # 'TNC', # Not precise enough
        # 'COBYLA', # Not precise enough
        'SLSQP', 'trust-constr',
        # 'dogleg',  # requires positive semi definite hessian, not the case here the problem is not convex
        'trust-ncg',
        'trust-exact',  # requires hessian
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
    """
    Adapated from: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """

    def fun(x): 
        return (x[0] - 1)**2 + (x[1] - 2.5)**2

    cons = {'type': 'ineq', 'fun': lambda x:
            np.array([1, -1, -1]) * x[0] + np.array([-2, -2, +2]) * x[1] + np.array([2, 6, 2])}

    bnds = ((0, None), (0, None))

    res = minimize(fun, np.array([2, 0]), method='SLSQP', bounds=bnds,
                   constraints=cons)

    assert_almost_equal(res.x, np.array([1.4, 1.7]), decimal=6)


def test_matrix_decomposition(shape=(10, 20), inner_shape=3, method=None):
    U = random((shape[0], inner_shape))
    V = random((inner_shape, shape[1]))
    prod = U@V

    def model(U=None, V=None):
        return tf.reduce_mean((U@V-tf.constant(prod, dtype=tf.float32))**2)

    def model_torch(smv=None, smp=None):
        return ((smv[:, None, :, None]*smp[None, :, None, :]-torch.tensor(Z, dtype=torch.float32))**2).mean()

    x0 = {'U': -random((shape[0], inner_shape)),
          'V': random((inner_shape, shape[1]))}

    tic = time()
    res = minimize(model, x0, method=method,
                   bounds={'U': (0, None), 'V': [
                       (0, None)]*inner_shape * shape[1]}
                   )
    print(method, time()-tic, res.fun)

    x0 = [random((shape[0], inner_shape)), random((inner_shape, shape[1]))]

    tic = time()
    res = minimize(model, x0, method=method)
    print(method, time()-tic, res.fun)


def n_knapsack(n_knapsacks=5,
               n_items=100,  # Should be divisible by n_knapsack
               n_weights_per_items=500,
               use_constraints=False,
               method='trust-constr',
               backend='tf'
               ):
    """
    Here we solve a continuous relaxation of the multiknapsack problem.

    """

    # Let's emulate the multiknapsack problem with random weights
    weights_ = random((n_weights_per_items, n_items))

    # We create knapsacks with attribution of the items to knapsacks [0,1,2,3,4] as:
    # [0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4]
    capacity_knapsacks = weights_.reshape(
        (n_weights_per_items, -1, n_knapsacks)).sum(-2)

    if backend == 'tf':
        weights_ = tf.constant(weights_, tf.float32)
        capacity_knapsacks_ = tf.constant(capacity_knapsacks, tf.float32)

        def func(W):
            # We use softmax to impose the constraint that the attribution of items to knapsacks should sum to one
            if use_constraints:
                W = tf.nn.softmax(W, 1)

            # We add a penalty only when the weights attribution sums higher than the knapsacks capacity.
            res = tf.nn.relu(weights_@W-capacity_knapsacks_)
            res = tf.reduce_mean(res**2)
            return res
        dev = None
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights_ = torch.tensor(weights_, dtype=torch.float32, device=dev)
        capacity_knapsacks_ = torch.tensor(
            capacity_knapsacks, dtype=torch.float32, device=dev)

        def func(W):
            # We use softmax to impose the constraint that the attribution of items to knapsacks should sum to one
            if use_constraints:
                W = torch.nn.functional.softmax(W, 1)

            # We add a penalty only when the weights attribution sums higher than the knapsacks capacity.
            res = torch.nn.functional.relu(weights_@W-capacity_knapsacks_)
            res = (res**2).mean()
            return res

    if use_constraints:
        if backend == 'tf':
            def eq_fun(W):
                return tf.reduce_sum(W, 1)-1
        else:
            def eq_fun(W):
                return W.sum(1)-1
        constraints = {
            'type': 'eq',
            'fun': eq_fun,
            'lb': 0,
            'ub': 0,
            'use_autograd': False
        }
    else:
        constraints = None

    Winit = np.zeros((n_items, n_knapsacks))
    res = minimize(func, Winit, tol=1e-8,
                   constraints=constraints,
                   bounds=(0, None),
                   method=method,
                   torch_device=dev,
                   backend=backend)
    return res


def test_nknpsack_tf_no_ctr():
    res = n_knapsack(n_knapsacks=2,
                     n_items=4,  # Should be divisible by n_knapsack
                     n_weights_per_items=20,
                     use_constraints=False,
                     method='trust-constr',
                     backend='tf'
                     )
    assert np.all(res.x.argmax(1) == np.array([0, 1, 0, 1]))


def test_nknpsack_tf_with_ctr():
    res = n_knapsack(n_knapsacks=2,
                     n_items=4,  # Should be divisible by n_knapsack
                     n_weights_per_items=20,
                     use_constraints=True,
                     method='trust-constr',
                     backend='tf'
                     )
    assert np.all(res.x.argmax(1) == np.array([0, 1, 0, 1]))


def test_nknpsack_tf_with_ctr_SLSQP():
    res = n_knapsack(n_knapsacks=2,
                     n_items=4,  # Should be divisible by n_knapsack
                     n_weights_per_items=20,
                     use_constraints=True,
                     method='SLSQP',
                     backend='tf'
                     )
    assert np.all(res.x.argmax(1) == np.array([0, 1, 0, 1]))


def test_nknpsack_torch_with_ctr():
    res = n_knapsack(n_knapsacks=2,
                     n_items=4,  # Should be divisible by n_knapsack
                     n_weights_per_items=20,
                     use_constraints=True,
                     method='trust-constr',
                     backend='torch'
                     )
    assert np.all(res.x.argmax(1) == np.array([0, 1, 0, 1]))


def test_nknpsack_torch_with_ctr_SLSQP():
    res = n_knapsack(n_knapsacks=2,
                     n_items=4,  # Should be divisible by n_knapsack
                     n_weights_per_items=20,
                     use_constraints=True,
                     method='SLSQP',
                     backend='torch'
                     )
    assert np.all(res.x.argmax(1) == np.array([0, 1, 0, 1]))


def test_keras_model_regression():
    # Prepares data
    X = np.random.random((200, 2))
    y = X[:, :1]*2+X[:, 1:]*0.4-1

    # Creates model
    model = keras.Sequential([keras.Input(shape=2),
                              layers.Dense(1)])

    # Transforms model into a function of its parameter
    func, params, names = tf_function_factory(model, tf.keras.losses.MSE, X, y)

    # Minimization
    res = minimize(func, params, method='trust-constr')

    assert_almost_equal(res.x[-1], -1, decimal=5)


def test_torch_model_regression():
    # Prepares data
    X = np.random.random((200, 2))
    y = X[:, :1]*2+X[:, 1:]*0.4-1

    # Creates model
    model = nn.Sequential(nn.Linear(2, 1))

    # Transforms model into a function of its parameter
    func, params, names = torch_function_factory(model, nn.MSELoss(), X, y)

    # Minimization
    res = minimize(func, params, method='trust-constr', backend='torch')

    assert_almost_equal(res.x[-1], -1, decimal=5)
