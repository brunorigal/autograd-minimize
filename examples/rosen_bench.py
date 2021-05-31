from time import time

import numpy as np
import tensorflow as tf
from autograd_minimize import minimize
import pandas as pd


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

    results = {}
    for method in [
        'Nelder-Mead', 'Powell', 'CG', 'BFGS',
        'Newton-CG',
        'L-BFGS-B',
        'TNC',
        'COBYLA',
        'SLSQP', 'trust-constr',
        'dogleg',  # requires positive semi definite hessian, not the case here the problem is not convex
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
        results[method] = {
            'time': time()-tic, 'error': np.mean(np.abs(res.x-1))}

    return results


if __name__ == '__main__':
    results = rosen_tst('tf')
    print(pd.DataFrame(results).T)
