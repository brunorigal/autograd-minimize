import numpy as np
import jax
from .base_wrapper import BaseWrapper
import jax.numpy as np
import numpy as onp


class JaxWrapper(BaseWrapper):
    def __init__(self, func, precision: str = "float32"):
        self.func = func

        if precision == "float32":
            self.precision = np.float32
        elif precision == "float64":
            self.precision = np.float64
        else:
            raise ValueError

    def get_value_and_grad(self, input_var):
        assert "shapes" in dir(self), "You must first call get input to define the tensors shapes."
        input_var_ = self._unconcat(np.array(input_var, dtype=self.precision), self.shapes)

        value, grads = self._get_value_and_grad(input_var_)

        return [
            onp.array(value).astype(onp.float64),
            onp.array(self._concat(grads)[0]).astype(onp.float64),
        ]

    def get_hvp(self, input_var, vector):
        assert "shapes" in dir(self), "You must first call get input to define the tensors shapes."
        input_var_ = self._unconcat(np.array(input_var, dtype=self.precision), self.shapes)
        vector_ = self._unconcat(np.array(vector, dtype=self.precision), self.shapes)

        res = self._get_hvp_tf(input_var_, vector_)
        return onp.array(self._concat(res)[0]).astype(onp.float64)

    def get_hess(self, input_var):
        assert "shapes" in dir(self), "You must first call get input to define the tensors shapes."
        input_var_ = np.array(input_var, dtype=self.precision)
        hess = onp.array(self._get_hess(input_var_)).astype(onp.float64)

        return hess

    def _get_hess(self, input_var):
        return jax.hessian(self._eval_func)(self._unconcat(input_var, self.shapes))

    def _get_value_and_grad(self, input_var):
        val_grad = jax.value_and_grad(self._eval_func)
        return val_grad(input_var)

    def _get_hvp_tf(self, input_var, vector):
        return hvp_fwd_rev(self._eval_func, input_var, vector)

    def get_ctr_jac(self, input_var):
        assert "shapes" in dir(self), "You must first call get input to define the tensors shapes."
        input_var_ = self._unconcat(np.array(input_var, dtype=self.precision), self.shapes)

        jac = self._get_ctr_jac(input_var_)

        return onp.array(jac).reshape((-1, self.var_num)).astype(onp.float64)

    def _get_ctr_jac(self, input_var):
        return jax.jacfwd(self._eval_ctr_func)(input_var)

    def _reshape(self, t, sh):
        if isinstance(t, onp.ndarray) or isinstance(t, np.ndarray):
            return np.reshape(t, sh)
        else:
            raise NotImplementedError

    def _tconcat(self, t_list, dim=0):
        if isinstance(t_list[0], onp.ndarray) or isinstance(t_list[0], np.ndarray):
            return np.concatenate(t_list, dim)
        else:
            raise NotImplementedError

    def _gather(self, t, i, j):
        if isinstance(t, onp.ndarray) or isinstance(t, np.ndarray):
            return t[i:j]
        elif i + 1 == j:
            return t
        else:
            raise NotImplementedError


# from: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#hessian-vector-products-using-both-forward-and-reverse-mode


# reverse-mode
def hvp(f, x, v):
    return jax.grad(lambda x: np.vdot(jax.grad(f)(x), v))(x)


# forward-over-reverse
def hvp_fwd_rev(f, primals, tangents):
    return jax.jvp(jax.grad(f), [primals], [tangents])[1]


# reverse-over-forward
def hvp_revfwd(f, primals, tangents):
    g = lambda primals: jax.jvp(f, [primals], [tangents])[1]
    return jax.grad(g)(primals)
