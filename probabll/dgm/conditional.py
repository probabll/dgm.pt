import torch
from torch.distributions import Distribution

from probabll.dgm import parameterize_conditional
from probabll.dgm.conditioners import Conditioner, MADEConditioner

        
class ConditionalLayer(torch.nn.Module):
    """
    Parameterises a torch.distributions.Distribution object.
    
    General idea:
    
        forward(inputs):
            outputs = conditioner(inputs)
            return parameterize_conditional(distribution_type, outputs, event_size)
    """
    
    def __init__(self, event_size: int, dist_type: type, conditioner: Conditioner):
        super(ConditionalLayer, self).__init__() 
        self.conditioner = conditioner
        self.dist_type = dist_type
        self.event_size = event_size
       
    def forward(self, inputs, **kwargs) -> Distribution:
        """
        :param inputs: [..., input_size]
        :return: a parameterized Distribution
        """
        outputs = self.conditioner(inputs, **kwargs)
        return parameterize_conditional(self.dist_type, outputs, self.event_size)

