import tensorflow as tf
import numpy as np
from numpy.random import random
from .base_wrapper import concat_, unconcat_, BaseWrapper

class TfWrapper(BaseWrapper):
    def __init__(self, func, precision='float32'):
        self.func = func

        if precision == 'float32':
            self.precision = tf.float32
        elif precision == 'float64':
            self.precision = tf.float64
        else:
            raise ValueError

    def get_input(self, input_var):
        input_, self.shapes = concat_(input_var)
        return input_

    def get_output(self, output_var):
        assert 'shapes' in dir(self), 'You must first call get input to define the tensors shapes.'
        output_var_ = unconcat_(output_var, self.shapes)
        return output_var_

    def get_bounds(self, bounds):
        return bounds

    def get_constraints(self, constraints):
        return constraints

    def get_value_and_grad(self, input_var):
        assert 'shapes' in dir(self), 'You must first call get input to define the tensors shapes.'
        input_var_ = unconcat_(tf.constant(
            input_var, dtype=self.precision), self.shapes)
        value, grads = self._get_value_and_grad_tf(input_var_)

        return [value.numpy().astype(np.float64), concat_(grads)[0].numpy().astype(np.float64)]

    def get_hvp(self, input_var, vector):
        assert 'shapes' in dir(self), 'You must first call get input to define the tensors shapes.'
        input_var_ = unconcat_(tf.constant(
            input_var, dtype=self.precision), self.shapes)
        vector_ = unconcat_(tf.constant(
            vector, dtype=self.precision), self.shapes)

        res = self._get_hvp_tf(input_var_, vector_)
        return concat_(res)[0].numpy().astype(np.float64)

    def get_hess(self, input_var):
        assert 'shapes' in dir(self), 'You must first call get input to define the tensors shapes.'
        input_var_ = tf.constant(input_var, dtype=self.precision)
        hess = self._get_hess(input_var_).numpy().astype(np.float64)

        return hess

    @tf.function
    def _get_hess(self, input_var):
        loss = self._eval_func(unconcat_(input_var, self.shapes))
        
        return tf.hessians(loss, input_var)[0]

    def _eval_func(self, input_var):
        if isinstance(input_var, dict):
            loss = self.func(**input_var)
        elif isinstance(input_var, list) or isinstance(input_var, tuple):
            loss = self.func(*input_var)
        else:
            loss = self.func(input_var)
        return loss

    @tf.function
    def _get_value_and_grad_tf(self, input_var):
        with tf.GradientTape() as tape:
            tape.watch(input_var)
            loss = self._eval_func(input_var)

        grad = tape.gradient(loss, input_var)
        return loss, grad

    @tf.function
    def _get_hvp_tf(self, input_var, vector):
        with tf.GradientTape() as outer_tape:
            outer_tape.watch(input_var)
            with tf.GradientTape() as inner_tape:
                inner_tape.watch(input_var)
                loss = self._eval_func(input_var)

            grads = inner_tape.gradient(loss, input_var)

        hvp = outer_tape.gradient(grads, input_var, output_gradients=vector)
        return hvp
