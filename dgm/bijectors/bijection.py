import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.distributions.utils import probs_to_logits, logits_to_probs


EPS = 1e-4


class Bijection(nn.Module):
    """
    An invertible transformation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, context):
        """
        :param inputs: [..., D]
        :param context: [..., H] or None
        :return: [..., D] outputs and [..., D] log determinant of Jacobian
        """
        raise NotImplementedError("Implement me!")

    def inverse(self, outputs, context):
        """
        :param outputs: [..., D]
        :param context: [..., H] or None
        :return: [..., D] inputs and [..., D] log determinant of Jacobian of inverse transform
        """
        raise NotImplementedError("Implement me!")


class SigmoidTransformer(Bijection):
    """
    Maps inputs from R to (0, 1) using a sigmoid.
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, context=None):
        log_p, log_q = - F.softplus(-inputs), - F.softplus(inputs)
        outputs = torch.sigmoid(inputs)
        return outputs, log_p + log_q

    def inverse(self, outputs, context=None):
        inputs = probs_to_logits(outputs, is_binary=True)  # stable implementation of inverse sigmoid
        log_p, log_q = - F.softplus(-inputs), - F.softplus(inputs)
        return inputs, - log_p - log_q


class FlipUnits(Bijection):

    def __init__(self, units):
        super().__init__()
        self.ids = torch.arange(units - 1, -1, -1).long()

    def forward(self, inputs, context=None):
        return inputs[:, self.ids], torch.zeros_like(inputs)

    def inverse(self, outputs, context=None):
        return outputs[:, self.ids], torch.zeros_like(outputs)



class StackedBijections(Bijection):

    def __init__(self, bijections: list):
        super().__init__()
        self.flows = nn.ModuleList(bijections)

    def forward(self, data_sample, context=None):
        """
        Example
        L = 2  (depth)
        x0 ~ data sample
            x1 = u0(x0) + s0(x0) * x0
            x2 = u1(x1) + s1(x1) * x1
        x2 ~ base sample
        :param data_sample: [..., d_i]
        :param context: [..., d_c]
        :return: [..., d_i]
        """
        outputs = data_sample
        log_det_jac = 0.

        for flow in self.flows:
            # Conditioner: predicts parameters of linear layer
            outputs, ldj = flow(outputs, context)
            log_det_jac = log_det_jac + ldj

        return outputs, log_det_jac

    def inverse(self, base_sample, context=None):
        """
        :param x: [..., d_i]
        :param context: [..., d_c]
        :return: [..., d_i]
        """
        outputs = base_sample
        log_det_jac = 0.
        for l, flow in enumerate(reversed(self.flows)):
            outputs, ldj = flow.inverse(outputs, context)
            log_det_jac = log_det_jac + ldj
        return outputs, log_det_jac



