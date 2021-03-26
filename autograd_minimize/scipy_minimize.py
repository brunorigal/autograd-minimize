from .tf_wrapper import TfWrapper
from .torch_wrapper import TorchWrapper
import scipy.optimize as sopt


def minimize(fun, x0, backend='tf', precision='float32', method=None, bounds=None, constraints=(), tol=None, callback=None, options=None):
    """wrapper around the minimize function of scipy which includes an automatic computation of gradients. 

    :param fun: function to be minimized, its signature can be a tensor, a list of tensors or a dict of tensors.
    :type fun: tensorflow of torch function

    :param x0: input to the function, it must match the signature of the function.
    :type x0: np.ndarray, list of arrays or dict of arrays. 

    :param backend: only tf is supported right now, defaults to 'tf'
    :type backend: str, optional

    :param precision: one of 'float32' or 'float64', defaults to 'float32'
    :type precision: str, optional

    :param method: method used by the optimizer, defaults to None
    :type method: str, optional

    :param bounds: [description], defaults to None
    :type bounds: [type], optional

    :param constraints: [description], defaults to ()
    :type constraints: tuple, optional

    :param tol: [description], defaults to None
    :type tol: [type], optional

    :param callback: [description], defaults to None
    :type callback: [type], optional

    :param options: [description], defaults to None
    :type options: [type], optional

    :return: dict of optimization results
    :rtype: dict
    """

    if backend == 'tf':
        wrapper = TfWrapper(fun, precision=precision)
    elif backend == 'torch':
        wrapper = TorchWrapper(fun, precision=precision)
    else:
        raise NotImplementedError

    optim_res = sopt.minimize(wrapper.get_value_and_grad,
                              wrapper.get_input(x0), method=method, jac=True,
                              hessp=wrapper.get_hvp if method in ['Newton-CG', 'trust-ncg',
                                                                  'trust-krylov', 'trust-constr']
                                                                else None,
                              hess=wrapper.get_hess if method in [
                                  'dogleg', 'trust-exact'] else None,
                              bounds=wrapper.get_bounds(bounds),
                              constraints=wrapper.get_constraints(constraints),
                              tol=tol, callback=callback, options=options)

    optim_res.x = wrapper.get_output(optim_res.x)

    if 'jac' in optim_res.keys():
        optim_res.jac = wrapper.get_output(optim_res.jac)

    return optim_res
