from time import time

import tensorflow as tf
import torch
from autograd_minimize import minimize
from numpy.random import random
import pandas as pd

results = {}


def test_matrix_decomposition(shape=(5, 10), inner_shape=3, backend='torch'):

    for method in [
        'Nelder-Mead',
        'Powell',
        'CG', 'BFGS',
        'Newton-CG',
        'L-BFGS-B',
        'TNC',
        'COBYLA',
        'SLSQP', 'trust-constr',
        # 'dogleg',  # requires positive semi definite hessian, not the case here the problem is not convex
        'trust-ncg',
        'trust-exact',
        'trust-krylov'
    ]:
        U = random((shape[0], inner_shape))
        V = random((inner_shape, shape[1]))
        prod = U@V

        def model_tf(U=None, V=None):
            return tf.reduce_mean((U@V-tf.constant(prod, dtype=tf.float32))**2)

        def model_torch(U=None, V=None):
            return ((U@V-torch.tensor(prod, dtype=torch.float32))**2).mean()

        model = model_tf if backend == 'tf' else model_torch

        x0 = {'U': -random((shape[0], inner_shape)),
              'V': random((inner_shape, shape[1]))}

        tic = time()
        res = minimize(model, x0, method=method,
                       # bounds={'U': (0, None), 'V': [(0, None)]*inner_shape* shape[1]}
                       backend=backend,
                       )

        toc = time()-tic
        results[method] = {'time': toc, 'mse': res.fun}

    return results


if __name__ == '__main__':
    results = test_matrix_decomposition()
    print(pd.DataFrame(results).T)
