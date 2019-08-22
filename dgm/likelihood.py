import torch
from torch.distributions import Distribution
from dgm import parameterize_conditional
from dgm.conditional import Conditioner, ConditionalLayer

class LikelihoodLayer(ConditionalLayer):
    """
    Likelihood layers add a sampling (in data space) functionality to a ConditionalLayer.
    """
    
    def __init__(self, event_size: int, dist_type: type, conditioner: Conditioner):
        """
        :param event_size: dimensionality of the random variable
        :param dist_type: a subclass of torch.distributions.Distribution
        :param conditioner: a NN architecture that maps inputs (conditioning context)
            to parameters of the given distribution
        """
        super(LikelihoodLayer, self).__init__(event_size, dist_type, conditioner)

    def sample(self, inputs, **kwargs):
        pass

    
class FullyFactorizedLikelihood(LikelihoodLayer):
    """
    Meant for models of the kind P(x|z) = \prod_d P(x_d|z)
    
    The argument `inputs` in forward corresponds to z.
    """
    
    def __init__(self, event_size: int, dist_type: type, conditioner: Conditioner):
        super(FullyFactorizedLikelihood, self).__init__(event_size, dist_type, conditioner)
        
    def sample(self, inputs, **kwargs):
        with torch.no_grad():
            p = self(inputs)
            outputs = p.sample()
            return outputs
        

class AutoregressiveLikelihood(LikelihoodLayer):
    """
    Meant for models of the kind P(x|z) = \prod_d P(x_d|x_{<d}, z)
    
    The argument `inputs` in forward corresponds to [x, z] and conditioners use that to, 
        for example, condition on x_{<d} and z.
    """
    
    def __init__(self,
                 event_size: int, dist_type: type, conditioner: Conditioner):
        super(AutoregressiveLikelihood, self).__init__(event_size, dist_type, conditioner)         

    def sample(self, inputs, outcome, start_from=0):
        # Make a copy of inputs and outcome so we can edit in-place
        inputs = torch.zeros_like(inputs) + inputs
        outcome = torch.zeros_like(outcome) + outcome
        # condition on x_{<d}
        inputs[...,:outcome.size(-1)] = outcome
        with torch.no_grad():
            for d in range(start_from, self.event_size): 
                # parameterize a conditional using x_{<d}
                p = self(inputs)
                # sample X_d|x_{<d}
                outcome[...,d] = p.sample()[...,d]
            return outcome        
        
    
