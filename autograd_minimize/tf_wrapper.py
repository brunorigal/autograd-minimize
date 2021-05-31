import tensorflow as tf
import numpy as np
from numpy.random import random
from .base_wrapper import BaseWrapper
from tensorflow.python.eager import forwardprop


class TfWrapper(BaseWrapper):
    def __init__(self, func, precision='float32', hvp_type='back_over_back_hvp'):
        self.func = func
        if 'is_keras_functional_model' not in dir(func):
            self.keras_model = False
        else:
            self.keras_model = func.is_keras_functional_model

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
        watch_var = self.func.trainable_variables if self.keras_model else input_var
        with tf.GradientTape() as tape:
            tape.watch(watch_var)
            loss = self._eval_func(input_var)

        grad = tape.gradient(loss, watch_var)
        return loss, grad

    @tf.function
    def _get_hvp_tf(self, input_var, vector):
        watch_var = self.func.trainable_variables if self.keras_model else input_var
        return self.hvp_func(self._eval_func, input_var, watch_var, vector)

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
        elif i+1 == j:
            return t
        else:
            raise NotImplementedError


# All hvp functions are copied from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/eager/benchmarks/resnet50/hvp_test.py
def _forward_over_back_hvp(func, input_var, watch_var, vector):
    with forwardprop.ForwardAccumulator(watch_var, vector) as acc:
        with tf.GradientTape() as grad_tape:
            grad_tape.watch(watch_var)
            loss = func(input_var)
        grads = grad_tape.gradient(loss, watch_var)

    return acc.jvp(grads)


def _back_over_forward_hvp(func, input_var, watch_var, vector):
    with tf.GradientTape() as grad_tape:
        grad_tape.watch(watch_var)
        with forwardprop.ForwardAccumulator(
                watch_var, vector) as acc:
            loss = func(input_var)
    return grad_tape.gradient(acc.jvp(loss), watch_var)


def _tf_gradients_forward_over_back_hvp(func, input_var, watch_var, vector):
    with tf.GradientTape() as grad_tape:
        grad_tape.watch(input_var)
        loss = func(input_var)

    variables = input_var
    grads = grad_tape.gradient(loss, variables)
    helpers = tf.nest.map_structure(tf.ones_like, grads)
    transposing = tf.gradients(grads, variables, helpers)
    return tf.gradients(transposing, helpers, vector)


def _back_over_back_hvp(func, input_var, watch_var, vector):
    with tf.GradientTape() as outer_tape:
        outer_tape.watch(watch_var)
        with tf.GradientTape() as inner_tape:
            inner_tape.watch(watch_var)
            loss = func(input_var)
        grads = inner_tape.gradient(loss, watch_var)

    return outer_tape.gradient(grads, watch_var, output_gradients=vector)


def tf_function_factory(model, loss, train_x, train_y):
    """
    A factory to create a function of the keras parameter model.

    The code is adapted from : https://gist.github.com/piyueh/712ec7d4540489aad2dcfb80f9a54993

    :param model: keras model
    :type model: tf.keras.Model]
    :param loss: a function with signature loss_value = loss(pred_y, true_y).
    :type loss: function
    :param train_x: dataset used as input of the model
    :type train_x: np.ndarray
    :param train_y: dataset used as   ground truth input of the loss
    :type train_y: np.ndarray
    :return: (function of the parameters, list of parameters, names of parameters)
    :rtype: tuple
    """

    # now create a function that will be returned by this factory
    def func(*params):
        # name2pos = {var.name: i for i, var in enumerate(model.trainable_variables)}
        # update the parameters in the model
        # for name, param in params.items():
        #     model.trainable_variables[name2pos[name]].assign(param)
        for i, param in enumerate(params):
            model.trainable_variables[i].assign(param)

        # calculate the loss
        loss_value = loss(model(train_x, training=True), train_y)

        return tf.reduce_mean(loss_value)

    func.trainable_variables = model.trainable_variables
    func.is_keras_functional_model = True

    params = [var.numpy() for var in model.trainable_variables]
    # params = {var.name: var.numpy() for var in model.trainable_variables}
    names = [var.name for var in model.trainable_variables]
    return func, params, names
