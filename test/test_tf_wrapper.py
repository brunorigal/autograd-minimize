import pdb
import tensorflow as tf
import numpy as np
from autograd_minimize import minimize
from time import time
from numpy.random import random


def model(x):
    return tf.reduce_sum(tf.square(tf.sinh(tf.cast(x, tf.float32))-tf.constant(2, dtype=tf.float32)))


x0 = np.ones(1000)


tic = time()
res = minimize(model, x0, method='Newton-CG')
print(time()-tic)

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

pdb.set_trace()
