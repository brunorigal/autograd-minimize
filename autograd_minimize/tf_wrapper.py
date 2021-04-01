import tensorflow as tf
import numpy as np
from numpy.random import random
from .base_wrapper import BaseWrapper
from tensorflow.python.eager import forwardprop


class TfWrapper(BaseWrapper):
    def __init__(self, func, precision='float32', hvp_type='back_over_back_hvp'):
        self.func = func

        if precision == 'float32':
            self.precision = tf.float32
        elif precision == 'float64':
            self.precision = tf.float64
        else:
            raise ValueError

        if hvp_type == 'forward_over_back':
            self.hvp_func = _forward_over_back_hvp
        elif hvp_type == 'back_over_forward':
            self.hvp_func = _back_over_forward_hvp
        elif hvp_type == 'tf_gradients_forward_over_back':
            self.hvp_func = _tf_gradients_forward_over_back_hvp
        elif hvp_type == 'back_over_back' or hvp_type is None:
            self.hvp_func = _back_over_back_hvp
        else:
            raise NotImplementedError

    def get_value_and_grad(self, input_var):
        assert 'shapes' in dir(
            self), 'You must first call get input to define the tensors shapes.'
        input_var_ = self._unconcat(tf.constant(
            input_var, dtype=self.precision), self.shapes)
        value, grads = self._get_value_and_grad_tf(input_var_)

        return [value.numpy().astype(np.float64), self._concat(grads)[0].numpy().astype(np.float64)]

    def get_hvp(self, input_var, vector):
        assert 'shapes' in dir(
            self), 'You must first call get input to define the tensors shapes.'
        input_var_ = self._unconcat(tf.constant(
            input_var, dtype=self.precision), self.shapes)
        vector_ = self._unconcat(tf.constant(
            vector, dtype=self.precision), self.shapes)

        res = self._get_hvp_tf(input_var_, vector_)
        return self._concat(res)[0].numpy().astype(np.float64)

    def get_hess(self, input_var):
        assert 'shapes' in dir(
            self), 'You must first call get input to define the tensors shapes.'
        input_var_ = tf.constant(input_var, dtype=self.precision)
        hess = self._get_hess(input_var_).numpy().astype(np.float64)

        return hess

    @tf.function
    def _get_hess(self, input_var):
        loss = self._eval_func(self._unconcat(input_var, self.shapes))

        return tf.hessians(loss, input_var)[0]

    @tf.function
    def _get_value_and_grad_tf(self, input_var):
        with tf.GradientTape() as tape:
            tape.watch(input_var)
            loss = self._eval_func(input_var)

        grad = tape.gradient(loss, input_var)
        return loss, grad

    @tf.function
    def _get_hvp_tf(self, input_var, vector):
        return self.hvp_func(self._eval_func, input_var, vector)

    def get_ctr_jac(self, input_var):
        assert 'shapes' in dir(
            self), 'You must first call get input to define the tensors shapes.'
        input_var_ = self._unconcat(tf.constant(
            input_var, dtype=self.precision), self.shapes)

        jac = self._get_ctr_jac(input_var_)

        return jac.numpy().reshape((-1, self.var_num)).astype(np.float64)

    @tf.function
    def _get_ctr_jac(self, input_var):
        with tf.GradientTape() as tape:
            tape.watch(input_var)
            ctr_val = self._eval_ctr_func(input_var)
        return tape.jacobian(ctr_val, input_var)

    def _reshape(self, t, sh):
        if isinstance(t, tf.Tensor):
            return tf.reshape(t, sh)
        elif isinstance(t, np.ndarray):
            return np.reshape(t, sh)
        else:
            raise NotImplementedError

    def _tconcat(self, t_list, dim=0):
        if isinstance(t_list[0], tf.Tensor):
            return tf.concat(t_list, dim)
        elif isinstance(t_list[0], np.ndarray):
            return np.concatenate(t_list, dim)
        else:
            raise NotImplementedError

    def _gather(self, t, i, j):
        if isinstance(t, tf.Tensor):
            return tf.gather(t, tf.range(i, j), 0)
        elif isinstance(t, np.ndarray):
            return t[i:j]
        else:
            raise NotImplementedError


# All hvp functions are copied from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/eager/benchmarks/resnet50/hvp_test.py
def _forward_over_back_hvp(func, input_var, vector):
    with forwardprop.ForwardAccumulator(input_var, vector) as acc:
        with tf.GradientTape() as grad_tape:
            grad_tape.watch(input_var)
            loss = func(input_var)
        grads = grad_tape.gradient(loss, input_var)

    return acc.jvp(grads)


def _back_over_forward_hvp(func, input_var, vector):
    with tf.GradientTape() as grad_tape:
        grad_tape.watch(input_var)
        with forwardprop.ForwardAccumulator(
                input_var, vector) as acc:
            loss = func(input_var)
    return grad_tape.gradient(acc.jvp(loss), input_var)


def _tf_gradients_forward_over_back_hvp(func, input_var, vector):
    with tf.GradientTape() as grad_tape:
        grad_tape.watch(input_var)
        loss = func(input_var)

    variables = input_var
    grads = grad_tape.gradient(loss, variables)
    helpers = tf.nest.map_structure(tf.ones_like, grads)
    transposing = tf.gradients(grads, variables, helpers)
    return tf.gradients(transposing, helpers, vector)


def _back_over_back_hvp(func, input_var, vector):
    with tf.GradientTape() as outer_tape:
        outer_tape.watch(input_var)
        with tf.GradientTape() as inner_tape:
            inner_tape.watch(input_var)
            loss = func(input_var)
        grads = inner_tape.gradient(loss, input_var)
    return outer_tape.gradient(
        grads, input_var, output_gradients=vector)
