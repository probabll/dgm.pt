import torch
from torch.distributions import Distribution

import probabll.dgm as dgm
from probabll.dgm import parameterize_conditional
from probabll.dgm.conditional import ConditionalLayer
from probabll.dgm.conditioners import Conditioner


class LikelihoodLayer(torch.nn.Module):
    """
    Likelihood layers are much like conditional layers, but they are bit more special.
    The typical likelihood looks like
        P(x_i|z,x_{<i}
    so we have two variables `inputs` for `z` and `history` for `x_{<i}`.
    """

    def __init__(self, event_size: int, dist_type: type, conditioner: Conditioner):
        """
        :param event_size: dimensionality of the random variable
        :param dist_type: a subclass of torch.distributions.Distribution
        :param conditioner: a NN architecture that maps inputs (conditioning context)
            to parameters of the given distribution
        """
        super(LikelihoodLayer, self).__init__()
        self.conditioner = conditioner
        self.dist_type = dist_type
        self.event_size = event_size

    def forward(self, inputs, history, **kwargs):
        pass

    def sample(self, inputs, history, **kwargs):
        pass


class FullyFactorizedLikelihood(LikelihoodLayer):
    """
    Meant for models of the kind P(x|z) = \prod_d P(x_d|z)
    """

    def __init__(self, event_size: int, dist_type: type, conditioner: Conditioner):
        super(FullyFactorizedLikelihood, self).__init__(
            event_size, dist_type, conditioner
        )

    def forward(self, inputs, history=None, **kwargs) -> Distribution:
        """
        Note that history is ignored.
        """
        outputs = self.conditioner(inputs, **kwargs)
        return parameterize_conditional(self.dist_type, outputs, self.event_size)

    def sample(self, inputs, history=None, start_from=0, **kwargs):
        """
        Note that history is ignored.
        """
        with torch.no_grad():
            p = self(inputs, **kwargs)
            outputs = p.sample()

        # replace the initial inputs if necessary
        if start_from > 0 and history is not None:
            outputs[..., :start_from] = history[..., :start_from]

        return outputs


class AutoregressiveLikelihood(LikelihoodLayer):
    """
    Meant for models of the kind P(x|z) = \prod_d P(x_d|x_{<d}, z)

    The argument `inputs` in forward corresponds to [x, z] and conditioners use that to,
        for example, condition on x_{<d} and z.
    """

    def __init__(self, event_size: int, dist_type: type, conditioner: Conditioner):
        super(AutoregressiveLikelihood, self).__init__(
            event_size, dist_type, conditioner
        )

    def forward(self, inputs, history, **kwargs) -> Distribution:
        """
        Note that history is required.
        """
        if inputs is None:
            h = history
        else:
            h = torch.cat([history, inputs], -1)
        outputs = self.conditioner(h, **kwargs)
        return parameterize_conditional(self.dist_type, outputs, self.event_size)

    def sample(self, inputs, history, start_from=0, **kwargs):
        # Make a copy of inputs and history so we can edit in-place
        outcome = torch.zeros_like(history) + history
        with torch.no_grad():
            for d in range(start_from, self.event_size):
                # parameterize a conditional using x_{<d}
                p = self(inputs, outcome, **kwargs)
                # sample X_d|x_{<d}
                outcome[..., d] = p.sample()[..., d]
            return outcome
