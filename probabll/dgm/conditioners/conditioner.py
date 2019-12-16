"""
API for conditioners, that is, architectures that convert inputs a fixed-dimension vector used to parameterise distributions.
"""
import torch


class Conditioner(torch.nn.Module):
    """
    A NN architecture maps a tensor of inputs to a tensor of outputs used to parameterize a distribution.
    """
    
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
    def forward(self, inputs, **kwargs):
        """
        :param inputs: [..., input_size]
        :return: [..., output_size]
        """
        raise NotImplementedError
        
