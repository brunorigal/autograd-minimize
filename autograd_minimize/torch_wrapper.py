import numpy as np
import torch
from .base_wrapper import BaseWrapper
from torch.autograd.functional import hvp, vhp, hessian


class TorchWrapper(BaseWrapper):
    def __init__(self, func, precision='float32', hvp_type='vhp'):
        self.func = func

        if precision == 'float32':
            self.precision = torch.float32
        elif precision == 'float64':
            self.precision = torch.float64
        else:
            raise ValueError

        self.hvp_func = hvp if hvp_type == 'hvp' else vhp

    def get_value_and_grad(self, input_var):
        assert 'shapes' in dir(
            self), 'You must first call get input to define the tensors shapes.'

        input_var_ = self._unconcat(torch.tensor(
            input_var, dtype=self.precision, requires_grad=True), self.shapes)

        loss = self._eval_func(input_var_)
        input_var_grad = input_var_.values() if isinstance(
            input_var_, dict) else input_var_
        grads = torch.autograd.grad(loss, input_var_grad)

        if isinstance(input_var_, dict):
            grads = {k: v for k, v in zip(input_var_.keys(), grads)}

        return [loss.cpu().detach().numpy().astype(np.float64),
                self._concat(grads)[0].cpu().detach().numpy().astype(np.float64)]

    def get_hvp(self, input_var, vector):
        assert 'shapes' in dir(
            self), 'You must first call get input to define the tensors shapes.'

        input_var_ = self._unconcat(torch.tensor(
            input_var, dtype=self.precision), self.shapes)
        vector_ = self._unconcat(torch.tensor(
            vector, dtype=self.precision), self.shapes)

        if isinstance(input_var_, dict):
            input_var_ = tuple(input_var_.values())
        if isinstance(vector_, dict):
            vector_ = tuple(vector_.values())

        loss, vhp_res = self.hvp_func(self.func, input_var_, v=vector_)

        return self._concat(vhp_res)[0].cpu().detach().numpy().astype(np.float64)

    def get_hess(self, input_var):
        assert 'shapes' in dir(
            self), 'You must first call get input to define the tensors shapes.'
        input_var_ = torch.tensor(input_var, dtype=self.precision)

        def func(inp):
            return self._eval_func(self._unconcat(inp, self.shapes))

        hess = hessian(func, input_var_, vectorize=False)

        return hess.cpu().detach().numpy().astype(np.float64)

    def get_ctr_jac(self, input_var):
        assert 'shapes' in dir(
            self), 'You must first call get input to define the tensors shapes.'

        input_var_ = self._unconcat(torch.tensor(
            input_var, dtype=self.precision, requires_grad=True), self.shapes)

        ctr_val = self._eval_ctr_func(input_var_)
        input_var_grad = input_var_.values() if isinstance(
            input_var_, dict) else input_var_
        grads = torch.autograd.grad(ctr_val, input_var_grad)

        return grads.cpu().detach().numpy().astype(np.float64)

    def _reshape(self, t, sh):
        if torch.is_tensor(t):
            return t.view(sh)
        elif isinstance(t, np.ndarray):
            return np.reshape(t, sh)
        else:
            raise NotImplementedError

    def _tconcat(self, t_list, dim=0):
        if torch.is_tensor(t_list[0]):
            return torch.cat(t_list, dim)
        elif isinstance(t_list[0], np.ndarray):
            return np.concatenate(t_list, dim)
        else:
            raise NotImplementedError

    def _gather(self, t, i, j):
        if isinstance(t, np.ndarray) or torch.is_tensor(t):
            return t[i:j]
        else:
            import pdb;pdb.set_trace()
            raise NotImplementedError
