import torch
import torch.nn.functional as F
from torch.distributions import Distribution
from dgm import parameterize_prior

    
class PriorLayer(torch.nn.Module):
    """Use this to parameterise priors with a specific batch_shape and in a certain device"""

    def __init__(self, event_shape, dist_type: type, params: list, dtype=torch.float32):
        super(PriorLayer, self).__init__()
        self.event_shape = event_shape
        self.dtype = dtype
        self.dist_type = dist_type
        self.params = params
        
    def forward(self, batch_shape, device) -> Distribution:
        return parameterize_prior(self.dist_type, batch_shape, self.event_shape, self.params, device, dtype=self.dtype)
        
        