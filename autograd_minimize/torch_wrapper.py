import numpy as np
import torch
from .base_wrapper import BaseWrapper
from torch.autograd.functional import hvp, vhp, hessian
from typing import List, Tuple, Dict, Union, Callable
from torch import nn, Tensor

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

        if isinstance(input_var_, list):
            input_var_ = tuple(input_var_)
        if isinstance(vector_, list):
            vector_ = tuple(vector_)

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
            raise NotImplementedError


def torch_function_factory(model, loss, train_x, train_y, precision='float32'):
    
    # named_params = {k: var.cpu().detach().numpy() for k, var in model.named_parameters()}
    params, names = extract_weights(model)
    
    prec_ = torch.float32 if precision == 'float32' else torch.float64
    train_x = torch.tensor(train_x, dtype=prec_)
    train_y = torch.tensor(train_y, dtype=prec_)

    def func(*new_params):
        load_weights(model, {k: v for  k, v in zip(names, new_params)})
        out = model(train_x)
        return loss(out, train_y)

    return func, [p.cpu().detach().numpy() for p in params]


#### Adapted from https://github.com/pytorch/pytorch/blob/21c04b4438a766cd998fddb42247d4eb2e010f9a/benchmarks/functional_autograd_benchmark/functional_autograd_benchmark.py

# Utilities to make nn.Module "functional"
# In particular the goal is to be able to provide a function that takes as input
# the parameters and evaluate the nn.Module using fixed inputs.
def _del_nested_attr(obj: nn.Module, names: List[str]) -> None:
    """
    Deletes the attribute specified by the given list of names.
    For example, to delete the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'])
    """
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        _del_nested_attr(getattr(obj, names[0]), names[1:])

def _set_nested_attr(obj: nn.Module, names: List[str], value: Tensor) -> None:
    """
    Set the attribute specified by the given list of names to value.
    For example, to set the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'], value)
    """
    if len(names) == 1:
        setattr(obj, names[0], value)
    else:
        _set_nested_attr(getattr(obj, names[0]), names[1:], value)

def extract_weights(mod: nn.Module) -> Tuple[Tuple[Tensor, ...], List[str]]:
    """
    This function removes all the Parameters from the model and
    return them as a tuple as well as their original attribute names.
    The weights must be re-loaded with `load_weights` before the model
    can be used again.
    Note that this function modifies the model in place and after this
    call, mod.parameters() will be empty.
    """
    orig_params = tuple(mod.parameters())
    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
        _del_nested_attr(mod, name.split("."))
        names.append(name)

    # Make params regular Tensors instead of nn.Parameter
    params = tuple(p.detach().requires_grad_() for p in orig_params)
    return params, names


def load_weights(mod: nn.Module, params: Dict[str, Tensor]) -> None:
    """
    Reload a set of weights so that `mod` can be used again to perform a forward pass.
    Note that the `params` are regular Tensors (that can have history) and so are left
    as Tensors. This means that mod.parameters() will still be empty after this call.
    """
    for name, p in params.items():
        _set_nested_attr(mod, name.split("."), p)