from abc import ABC, abstractmethod
import tensorflow as tf 
import torch
import numpy as np


class BaseWrapper(ABC):

    def get_input(self, input_var):
        self.input_type = type(input_var)
        assert self.input_type in [dict, list, np.ndarray], 'The initial input to your optimized function should be one of dict, list or np.ndarray'
        input_, self.shapes = concat_(input_var)
        self.var_num = input_.shape[0]
        return input_

    def get_output(self, output_var):
        assert 'shapes' in dir(self), 'You must first call get input to define the tensors shapes.'
        output_var_ = unconcat_(output_var, self.shapes)
        return output_var_

    def get_bounds(self, bounds):

        if bounds is not None:
            if isinstance(bounds, tuple):
                assert len(bounds)==2
                new_bounds = [bounds]*self.var_num

            elif isinstance(bounds, list):
                assert self.input_type == list 
                assert len(self.shapes)==len(bounds)
                new_bounds = []
                for sh, bounds_ in zip(self.shapes, bounds):
                    if isinstance(bounds_, tuple):
                        assert len(bounds_)==2
                        new_bounds+=[bounds_]*np.prod(sh, dtype=np.int32)
                    elif (isinstance(bounds_, list) or isinstance(bounds_, np.ndarray)):
                        assert np.prod(sh)==np.prod(bounds_.shape)
                        new_bounds+=np.concatenate(np.reshape(bounds_, -1)).tolist()
                    else:
                        raise TypeError

            elif isinstance(bounds, dict):
                assert self.input_type == dict 
                assert set(self.shapes.keys())==set(bounds.keys())

                new_bounds = []
                for k in self.shapes.keys():
                    sh = self.shapes[k]
                    if isinstance(bounds[k], tuple):
                        assert len(bounds[k])==2
                        new_bounds+=[bounds[k]]*np.prod(sh, dtype=np.int32)
                    elif (isinstance(bounds[k], list) or isinstance(bounds[k], np.ndarray)):
                        assert np.prod(sh)==np.prod(bounds[k].shape)
                        new_bounds+=np.concatenate(np.reshape(bounds[k], -1)).tolist()
                    else:
                        raise TypeError
        else:
            new_bounds = bounds
        return new_bounds

    def get_constraints(self, constraints):
        return constraints

    @abstractmethod
    def get_value_and_grad(self, input_var):
        return

    @abstractmethod
    def get_hvp(self, input_var, vector):
        return

    @abstractmethod
    def get_hess(self, input_var):
        return

    def _eval_func(self, input_var):
        if isinstance(input_var, dict):
            loss = self.func(**input_var)
        elif isinstance(input_var, list) or isinstance(input_var, tuple):
            loss = self.func(*input_var)
        else:
            loss = self.func(input_var)
        return loss



def reshape(t, sh):
    if isinstance(t, tf.Tensor):
        return tf.reshape(t, sh)
    elif torch.is_tensor(t):
        return t.view(sh)
    elif isinstance(t, np.ndarray):
        return np.reshape(t, sh)
    else:
        raise NotImplementedError

def concat(t_list, dim=0):
    if isinstance(t_list[0], tf.Tensor):
        return tf.concat(t_list, dim)
    elif torch.is_tensor(t_list[0]):
        return torch.cat(t_list, dim)
    elif isinstance(t_list[0], np.ndarray):
        return np.concatenate(t_list, dim)
    else:
        raise NotImplementedError

def gather(t, i, j):
    if isinstance(t, np.ndarray) or torch.is_tensor(t):
        return t[i:j]
    elif isinstance(t, tf.Tensor):
        return tf.gather(t, tf.range(i, j), 0)
    else:
        raise NotImplementedError


def concat_(ten_vals):
    ten = []
    if isinstance(ten_vals, dict):
        shapes = {}
        for k, t in ten_vals.items():
            if t is not None:
                shapes[k] = t.shape
                ten.append(reshape(t, [-1]))
        ten = concat(ten, 0)

    elif isinstance(ten_vals, list) or isinstance(ten_vals, tuple):
        shapes = []
        for t in ten_vals:
            if t is not None:
                shapes.append(t.shape)
                ten.append(reshape(t, [-1]))
        ten = concat(ten, 0)

    else:
        shapes = None
        ten = ten_vals

    return ten, shapes


def unconcat_(ten, shapes):
    current_ind = 0
    if isinstance(shapes, dict):
        ten_vals = {}
        for k, sh in shapes.items():
            next_ind = current_ind+np.prod(sh, dtype=np.int32)
            ten_vals[k] = reshape(gather(ten, current_ind, next_ind), sh)
            # if isinstance(ten, np.ndarray):
            #     ten_vals[k] = np.reshape(ten[current_ind:next_ind], sh)
            # else:
            #     ten_vals[k] = tf.reshape(
            #         tf.gather(ten, tf.range(current_ind, next_ind), 0), sh)

            current_ind = next_ind

    elif isinstance(shapes, list) or isinstance(shapes, tuple):
        ten_vals = []
        for sh in shapes:
            next_ind = current_ind+np.prod(sh, dtype=np.int32)
            ten_vals.append(reshape(gather(ten, current_ind, next_ind), sh))
            # if isinstance(ten, np.ndarray):
            #     ten_vals.append(np.reshape(ten[current_ind:next_ind], sh))
            # else:
            #     ten_vals.append(tf.reshape(
            #         tf.gather(ten, tf.range(current_ind, next_ind), 0), sh))

            current_ind = next_ind

    elif shapes is None:
        ten_vals = ten

    return ten_vals
