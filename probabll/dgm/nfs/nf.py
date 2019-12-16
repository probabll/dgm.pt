import torch
from torch.distributions import Distribution
from torch.distributions.kl import register_kl
from probabll.dgm import register_conditional_parameterization
from probabll.dgm.bijectors import Bijection


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

    def nf_rsample(self, sample_shape=torch.Size()):
        base_sample = self.base.rsample(sample_shape)
        data_sample, djac = self.flow.inverse(base_sample, self.context)
        return base_sample, data_sample, djac
    
    def rsample(self, sample_shape=torch.Size()):
        """Return data samples by sampling from the uniform base and transforming with the inverse rectified flow"""
        _, data_sample, _ = self.nf_rsample(sample_shape=sample_shape)
        return data_sample

    def sample(self, sample_shape=torch.Size()):
        """Sample as rsample but without gradients"""
        with torch.no_grad():
            return self.rsample(sample_shape)

    def log_prob(self, data_sample):
        base_sample, log_det_jac = self.flow(data_sample, self.context)
        return self.base.log_prob(base_sample) + log_det_jac

    def cdf(self, data_sample):
        base_sample, _ = self.flow(data_sample, self.context)
        return self.base.cdf(base_sample)


class IAF(NF):
    
    def __init__(self, base: Distribution, flow: Bijection, context=None, validate_args=None):
        """
        :param shape:
        :param flow:
        :param base: defaults to Normal(0, 1)
        :param context:
        :param validate_args:
        :param device:
        """
        super().__init__(base, flow, context, validate_args)

    def nf_rsample(self, sample_shape=torch.Size()):
        base_sample = self.base.rsample(sample_shape)
        data_sample, djac = self.flow(base_sample, self.context)
        return base_sample, data_sample, djac
    
    def rsample(self, sample_shape=torch.Size()):
        """Return data samples by sampling from the uniform base and transforming with the inverse rectified flow"""
        _, data_sample, _ = self.nf_rsample(sample_shape=sample_shape)
        return data_sample

    def sample(self, sample_shape=torch.Size()):
        """Sample as rsample but without gradients"""
        with torch.no_grad():
            return self.rsample(sample_shape)

    def log_prob(self, data_sample):
        base_sample, log_det_jac = self.flow.inverse(data_sample, self.context)
        return self.base.log_prob(base_sample) + log_det_jac

    def cdf(self, data_sample):
        base_sample, _ = self.flow.inverse(data_sample, self.context)
        return self.base.cdf(base_sample)


@register_kl(NF, Distribution)
def estimate_kl(q, p):
    eps, x, log_det_jac = q.nf_rsample()
    return q.base.log_prob(eps) + log_det_jac - p.log_prob(x)


@register_conditional_parameterization(NF)
def make_nf(inputs: dict, event_size):
    return NF(inputs['base'], inputs['bijector'], context=inputs.get('context', None))

@register_conditional_parameterization(IAF)
def make_nf(inputs: dict, event_size):
    return IAF(inputs['base'], inputs['bijector'], context=inputs.get('context', None))

