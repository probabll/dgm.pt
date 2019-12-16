"""
Helper code based on torch.distributions.register_kl to register a parameterization of
    torch.distributions.Distribution objects.

Prior parameterization differs from conditional parameterization in that priors are specified via a list of floats.
Conditionals are parameterized via a tensor.
"""

import torch
import torch.nn.functional as F
from functools import total_ordering
from torch.distributions import Distribution, Uniform, Bernoulli, Normal


_CONDITIONAL_REGISTRY = {}  # Source of truth mapping a few general (type, type) pairs to functions.
_CONDITIONAL_MEMOIZE = {}

_PRIOR_REGISTRY = {}  # Source of truth mapping a few general (type, type) pairs to functions.
_PRIOR_MEMOIZE = {}


def register_prior_parameterization(type_p):
    """
    Decorator to register a function that parameterizes a distribution of type_p.
    Usage::

        @register_parameterization(Normal)
        def make_normal(inputs, event_size):
            params = torch.split(inputs, event_size, -1)
            return Normal(loc=params[0], scale=F.softplus(params[1]))

    Lookup returns the most specific (type). If
    the match is ambiguous, a `RuntimeWarning` is raised.
    
    Args:
        type_p (type): A subclass of :class:`~torch.distributions.Distribution`.
    """
    if not isinstance(type_p, type) and issubclass(type_p, Distribution):
        raise TypeError('Expected type_p to be a Distribution subclass but got {}'.format(type_p))

    def decorator(fun):
        _PRIOR_REGISTRY[type_p] = fun
        _PRIOR_MEMOIZE.clear()  # reset since lookup order may have changed
        return fun

    return decorator

def register_conditional_parameterization(type_p):
    """
    Decorator to register a function that parameterizes a distribution of type_p.
    Usage::

        @register_parameterization(Normal)
        def make_normal(inputs, event_size):
            params = torch.split(inputs, event_size, -1)
            return Normal(loc=params[0], scale=F.softplus(params[1]))

    Lookup returns the most specific (type). If
    the match is ambiguous, a `RuntimeWarning` is raised.
    
    Args:
        type_p (type): A subclass of :class:`~torch.distributions.Distribution`.
    """
    if not isinstance(type_p, type) and issubclass(type_p, Distribution):
        raise TypeError('Expected type_p to be a Distribution subclass but got {}'.format(type_p))

    def decorator(fun):
        _CONDITIONAL_REGISTRY[type_p] = fun
        _CONDITIONAL_MEMOIZE.clear()  # reset since lookup order may have changed
        return fun

    return decorator


@total_ordering
class _Match(object):
    __slots__ = ['types']

    def __init__(self, *types):
        self.types = types

    def __eq__(self, other):
        return self.types == other.types

    def __le__(self, other):
        for x, y in zip(self.types, other.types):
            if not issubclass(x, y):
                return False
            if x is not y:
                break
        return True


def _dispatch_prior(type_p):
    """
    Find the most specific approximate match, assuming single inheritance.
    """
    fun = _PRIOR_REGISTRY.get(type_p, None)
    if fun is None:
        return NotImplemented
    return fun


def _dispatch_conditional(type_p):
    """
    Find the most specific approximate match, assuming single inheritance.
    """
    fun = _CONDITIONAL_REGISTRY.get(type_p, None)
    if fun is None:
        return NotImplemented
    return fun


def parameterize_prior(type_p, batch_shape, event_shape, params: list, device, dtype=torch.float32) -> Distribution:
    r"""
    Parameterizes a distribution of a given type.    

    Args:
        type_p (type): 
        inputs (Tensor): 
        event_size (int):

    Returns:
        Tensor: an instance of type_p 

    Raises:
        NotImplementedError: If the distribution types have not been registered via
            :meth:`register_parameterization`.
    """
    try:
        fun = _PRIOR_MEMOIZE[type_p]
    except KeyError:
        fun = _dispatch_prior(type_p)
        _PRIOR_MEMOIZE[type_p] = fun
    if fun is NotImplemented:
        raise NotImplementedError("I cannot find a parameterization for a prior of type %s" % type_p )
    if isinstance(batch_shape, int):
        batch_shape = [batch_shape]
    if isinstance(event_shape, int):
        event_shape = [event_shape]
    return fun(batch_shape, event_shape, params, device=device, dtype=dtype)


def parameterize_conditional(type_p, inputs, event_size) -> Distribution:
    r"""
    Parameterizes a distribution of a given type.    

    Args:
        type_p (type): 
        inputs (Tensor): 
        event_size (int):

    Returns:
        Tensor: an instance of type_p 

    Raises:
        NotImplementedError: If the distribution types have not been registered via
            :meth:`register_parameterization`.
    """
    try:
        fun = _CONDITIONAL_MEMOIZE[type_p]
    except KeyError:
        fun = _dispatch_conditional(type_p)
        _CONDITIONAL_MEMOIZE[type_p] = fun
    if fun is NotImplemented:
        raise NotImplementedError("I cannot find a parameterization for a conditional of type %s" % type_p )
    return fun(inputs, event_size)
                

# Let's register a few basic parameterization of priors and conditionals    

    
@register_prior_parameterization(Bernoulli)
def parameterize(batch_shape, event_shape, params, device, dtype):
    return Bernoulli(probs=torch.full(batch_shape + event_shape, params[0], device=device, dtype=dtype))


@register_prior_parameterization(Normal)
def parameterize(batch_shape, event_shape, params, device, dtype):
    p = Normal(
        loc=torch.full(batch_shape + event_shape, params[0], device=device, dtype=dtype),
        scale=torch.full(batch_shape + event_shape, params[1], device=device, dtype=dtype),
    )
    return p


@register_conditional_parameterization(Bernoulli)
def make_bernoulli(inputs, event_size):
    assert inputs.size(-1) == event_size, "Expected [...,%d] got [...,%d]" % (event_size, inputs.size(-1))
    return Bernoulli(logits=inputs)


@register_conditional_parameterization(Normal)
def make_gaussian(inputs, event_size):
    assert inputs.size(-1) == 2 * event_size, "Expected [...,%d] got [...,%d]" % (2 * event_size, inputs.size(-1))
    params = torch.split(inputs, event_size, -1)
    return Normal(loc=params[0], scale=F.softplus(params[1])) 



