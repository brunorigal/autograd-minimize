import scipy.optimize as sopt


def minimize(
    fun,
    x0,
    backend="tf",
    precision="float32",
    method=None,
    hvp_type=None,
    torch_device="cpu",
    bounds=None,
    constraints=None,
    tol=None,
    callback=None,
    options=None,
):
    """
    wrapper around the [minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
    function of scipy which includes an automatic computation of gradients,
    hessian vector product or hessian with tensorflow or torch backends.

    Args:
    * fun: function to be minimized, its signature can be a tensor, a list of tensors or a dict of tensors.

    * x0: input to the function, it must match the signature of the function.

    * backend: one of 'tf' or 'torch', defaults to 'tf'

    * precision: one of 'float32' or 'float64', defaults to 'float32'

    * method: method used by the optimizer, it should be one of:
        'Nelder-Mead',
        'Powell',
        'CG',
        'BFGS',
        'Newton-CG',
        'L-BFGS-B',
        'TNC',
        'COBYLA',
        'SLSQP',
        'trust-constr',
        'dogleg',  # requires positive semi definite hessian
        'trust-ncg',
        'trust-exact', # requires hessian
        'trust-krylov'
        , defaults to None

    * hvp_type: type of computation scheme for the hessian vector product
        for the torch backend it is one of hvp and vhp (vhp is faster according to the [doc](https://pytorch.org/docs/stable/autograd.html))
        for the tf backend it is one of 'forward_over_back', 'back_over_forward', 'tf_gradients_forward_over_back' and 'back_over_back'
        Some infos about the most interesting scheme are given [here](https://www.tensorflow.org/api_docs/python/tf/autodiff/ForwardAccumulator)
        , defaults to None

    * torch_device: device used by torch for the gradients computation,
        if the backend is not torch, this parameter is ignored, defaults to 'cpu'

    * bounds: Bounds on the input variables, only available for L-BFGS-B, TNC, SLSQP, Powell, and trust-constr methods.
        It can be:
        * a tuple (min, max), None indicates no bounds, in this case the same bound is applied to all variables.
        * An instance of the [Bounds](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.Bounds.html#scipy.optimize.Bounds) class, in this case the same bound is applied to all variables.
        * A numpy array of bounds (if the optimized function has a single numpy array as input)
        * A list or dict of bounds with the same format as the optimized function signature.
        , defaults to None

    * constraints: It has to be a dict with the following keys:
        * fun: a callable computing the constraint function
        * lb and ub: the lower and upper bounds, if equal, the constraint is an inequality, use np.inf if there is no upper bound. Only used if method is trust-constr.
        * type: 'eq' or 'ineq' only used if method is one of COBYLA, SLSQP.
        * keep_feasible: see [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.NonlinearConstraint.html#scipy.optimize.NonlinearConstraint)
        , defaults to None

    * tol: Tolerance for termination, defaults to None

    * callback: Called after each iteration, defaults to None

    * options: solver options, defaults to None

    """

    if backend == "tf":
        from .tf_wrapper import TfWrapper

        wrapper = TfWrapper(fun, precision=precision, hvp_type=hvp_type)
    elif backend == "torch":
        from .torch_wrapper import TorchWrapper

        wrapper = TorchWrapper(fun, precision=precision, hvp_type=hvp_type, device=torch_device)
    elif backend == "jax":
        from .jax_wrapper import JaxWrapper

        wrapper = JaxWrapper(fun, precision=precision)
    else:
        raise NotImplementedError

    if bounds is not None:
        assert method in [
            None,
            "L-BFGS-B",
            "TNC",
            "SLSQP",
            "Powell",
            "trust-constr",
        ], "bounds are only available for L-BFGS-B, TNC, SLSQP, Powell, trust-constr"

    if constraints is not None:
        assert method in [
            "COBYLA",
            "SLSQP",
            "trust-constr",
        ], "Constraints are only available for COBYLA, SLSQP and trust-constr"

    optim_res = sopt.minimize(
        wrapper.get_value_and_grad,
        wrapper.get_input(x0),
        method=method,
        jac=True,
        hessp=wrapper.get_hvp if method in ["Newton-CG", "trust-ncg", "trust-krylov", "trust-constr"] else None,
        hess=wrapper.get_hess if method in ["dogleg", "trust-exact"] else None,
        bounds=wrapper.get_bounds(bounds),
        constraints=wrapper.get_constraints(constraints, method),
        tol=tol,
        callback=callback,
        options=options,
    )

    optim_res.x = wrapper.get_output(optim_res.x)

    if "jac" in optim_res.keys() and len(optim_res.jac) > 0:
        try:
            optim_res.jac = wrapper.get_output(optim_res.jac[0])
        except Exception:
            pass

    return optim_res
