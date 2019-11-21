import torch
from torch.distributions import Distribution
from torch.distributions.kl import register_kl

from dgm import register_conditional_parameterization
from dgm.bijectors import Bijection


class NF(Distribution):
    
    def __init__(self, base: Distribution, flow: Bijection, context=None, validate_args=None):
        """
        :param shape:
        :param flow:
        :param base: defaults to Normal(0, 1)
        :param context:
        :param validate_args:
        :param device:
        """
        super().__init__(batch_shape=base.batch_shape, event_shape=base.event_shape, validate_args=validate_args)
        self.base = base
        self.flow = flow  
        self.base = base
        self.context = context

    def rsample(self, sample_shape=torch.Size()):
        """Return data samples by sampling from the uniform base and transforming with the inverse rectified flow"""
        base_sample = self.base.rsample(sample_shape)
        data_sample, _ = self.flow.inverse(base_sample, self.context)
        return data_sample

    def sample(self, sample_shape=torch.Size()):
        """Sample as rsample but without gradients"""
        with torch.no_grad():
            return self.rsample(sample_shape)

    def log_prob(self, data_sample):
        base_sample, log_det_jac = self.flow(data_sample, self.context)
        return self.base.log_prob(base_sample) + log_det_jac


@register_kl(NF, Distribution)
def estimate_kl(q, p):
    x = q.rsample()
    return q.log_prob(x) - p.log_prob(x)


@register_conditional_parameterization(NF)
def make_nf(inputs: dict, event_size):
    return NF(inputs['base'], inputs['bijector'], context=inputs.get('context', None))

