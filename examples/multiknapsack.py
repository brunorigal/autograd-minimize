import tensorflow as tf
import numpy as np
from numpy.random import random
from time import time
from autograd_minimize import minimize
import torch


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
    else:
        weights_ = torch.tensor(weights_, dtype=torch.float32)
        capacity_knapsacks_ = torch.tensor(
            capacity_knapsacks, dtype=torch.float32)

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
        constraints = ()

    Winit = np.zeros((n_items, n_knapsacks))
    res = minimize(func, Winit, tol=1e-8,
                   constraints=constraints,
                   bounds=(0, None),
                   method=method,
                   backend=backend)
    return res


if __name__ == '__main__':
    tic = time()
    res = n_knapsack(n_knapsacks=5,
                     n_items=100,  # Should be divisible by n_knapsack
                     n_weights_per_items=500,
                     use_constraints=True,
                     method='trust-constr',
                     backend='torch')
    print(time()-tic)
    # And here is the attribution result:
    print(res.x.argmax(1))
