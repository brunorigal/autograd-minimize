import tensorflow as tf
import numpy as np
import scipy.optimize as sopt
# https://stackoverflow.com/questions/59029854/use-scipy-optimizer-with-tensorflow-2-0-for-neural-network-training
def model(x):
    # return tf.reduce_sum(tf.square(tf.cast(x, tf.float32)-tf.constant(2, dtype=tf.float32)))
    return tf.reduce_sum(tf.square(tf.sinh(tf.cast(x, tf.float32))-tf.constant(2, dtype=tf.float32)))





@tf.function
def val_and_grad(x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = model(x)
    grad = tape.gradient(loss, x)
    return loss, grad

def func(x):   
    res = val_and_grad(tf.constant(x, dtype=tf.float32))
    return [vv.numpy().astype(np.float64)  for vv in res]

@tf.function
def _back_over_back_hvp(func, input_var, vector):
    with tf.GradientTape() as outer_tape:
        with tf.GradientTape() as inner_tape:
            loss = func(input_var)
        grads = inner_tape.gradient(loss, input_var)
    
    hvp = outer_tape.gradient(grads, input_var, output_gradients=vector)
    return hvp

def tf_wrapper(func, input_var, vector):

    res = _back_over_back_hvp(func, tf.Variable(input_var, dtype=tf.float32,trainable=True), tf.Variable(vector, dtype=tf.float32, trainable=True)).numpy().astype(np.float64)

    return res


class wrap_tf:
    def __init__(self, func, precision='float32'):
        self.func = func

        if precision=='float32':
            self.tf_prec = tf.float32
        elif precision=='float32':
            self.tf_prec = tf.float64 

    
    def get_value_and_grad(self, input_var):
        res = self._get_value_and_grad_tf(tf.constant(input_var, dtype=self.tf_prec))
        return [vv.numpy().astype(np.float64)  for vv in res]


    @tf.function
    def _get_value_and_grad_tf(self, input_var):
        with tf.GradientTape() as tape:
            tape.watch(input_var)
            loss = self.func(input_var)
        grad = tape.gradient(loss, input_var)
        return loss, grad

    @tf.function
    def _get_hvp_tf(self, input_var, vector):
        with tf.GradientTape() as outer_tape:
            with tf.GradientTape() as inner_tape:
                loss = self.func(input_var)
            grads = inner_tape.gradient(loss, input_var)
        
        hvp = outer_tape.gradient(grads, input_var, output_gradients=vector)
        return hvp

    def get_hvp(self, input_var, vector):

        
from time import time
tic = time()
resdd= sopt.minimize(fun=func, x0=np.ones(1000),
                                    jac=True, hessp=lambda input_var, vector: tf_wrapper(model, input_var, vector), 
                                    # method='L-BFGS-B'
                                    method='Newton-CG'
                                    )
print(time()-tic)
print(np.mean(np.abs(resdd.x-2)))
import pdb;pdb.set_trace()

res = ['Newton-CG', 'dogleg', 'trust-ncg', 'trust-krylov', 'trust-exact', 'trust-constr']


from numpy.random import random

smp = random((5,5))
smv=random((3,3))

Z = smp[:,None, :,None]*smp[None,:,None,:]


def model(smw=None, smp=None):
    return tf.reduce_mean((smw[:,None,:,None]*smp[None,:,None,:]-tf.constant(Z, dtype=tf.float32))**2)

@tf.function
def val_and_grad(x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = model(x)
    grad = tape.gradient(loss, x)
    return loss, grad

def func(x):   
    res = val_and_grad(tf.constant(x, dtype=tf.float32))
    return [vv.numpy().astype(np.float64)  for vv in res]


import pdb;pdb.set_trace()