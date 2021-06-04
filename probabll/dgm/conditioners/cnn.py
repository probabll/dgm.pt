"""
Conditioners based on CNNs
"""
import torch

from probabll.dgm.nn import GatedConv2d, GatedConvTranspose2d
from .conditioner import Conditioner


class Conv2DConditioner(Conditioner):
    """
    Conditions on input shaped as [batch_size, width, height] using 2D convolutions
    """
    
    def __init__(self, input_size, output_size, width, height, 
                 output_channels, last_kernel_size, context_size=0, dropout=0):   
        """
        :param input_size: number of units in the input 
        :param output_size: number of outputs         
        :param width: for 2D convolutions we shape the inputs to [batch_size, width, height]
            (context units are not conditioned on via convolutions)
        :param height: for 2D convolutions we shape the inputs to [batch_size, width, height]
            (context units are not conditioned on via convolutions)
        :param output_channels: 
        :param last_kernel_size:
        :param context_size: (optional) number of inputs to be given special treatment
            these are always assumed to be the rightmost units
        """
        super().__init__(input_size, output_size)
        input_channels = (input_size - context_size) // (width * height)
        # Architecture from https://github.com/riannevdberg/sylvester-flows/blob/master/models/VAE.py
        # TODO: generalise these parameters
        self.cnn = torch.nn.Sequential(
            GatedConv2d(input_channels, 32, kernel_size=5, stride=1, padding=2),
            torch.nn.Dropout(dropout),
            GatedConv2d(32, 32, kernel_size=5, stride=2, padding=2),
            torch.nn.Dropout(dropout),
            GatedConv2d(32, 64, kernel_size=5, stride=1, padding=2),
            torch.nn.Dropout(dropout),
            GatedConv2d(64, 64, kernel_size=5, stride=2, padding=2),
            torch.nn.Dropout(dropout),
            GatedConv2d(64, 64, kernel_size=5, stride=1, padding=2),
            torch.nn.Dropout(dropout),
            GatedConv2d(64, output_channels, kernel_size=last_kernel_size, stride=1, padding=0)            
        )           
        if context_size > 0:            
            self.bridge = torch.nn.Sequential(
                torch.nn.Linear(output_channels + context_size, output_channels), 
                torch.nn.ReLU()
            )
        else:
            self.bridge = torch.nn.Identity()
        self.output_layer = torch.nn.Linear(output_channels, output_size)
        self.input_channels = input_channels
        self.width = width
        self.height = height
        self.context_size = context_size
    
    def forward(self, inputs, **kwargs):
        if self.context_size > 0:  # some of the units are considered "context"       
            inputs, context = torch.split(inputs, self.input_size - self.context_size, -1)            
        else:
            context = None            
        # [B, H*W] -> [B, 1, W, H]
        inputs = inputs.reshape(inputs.size(0), self.input_channels, self.height, self.width).permute(0, 1, 3, 2)
        h = self.cnn(inputs)
        h = h.view(inputs.size(0), -1)
        if context is not None:
            h = self.bridge(torch.cat([h, context], -1))            
        return self.output_layer(h)
    
    
class TransposedConv2DConditioner(Conditioner):
    """
    Paramaterises models of the kind P(x|z) = \prod_d P(x_d|z)    
    """
    
    def __init__(self, input_size: int, output_size: int, 
                 input_channels: int, output_channels: int, last_kernel_size: int, context_size=0, dropout=0):
        """
        :param input_size: number of units in the input 
        :param output_size: number of outputs         
        :param input_channels: 
        :param output_channels:
        :param last_kernel_size:
        :param context_size: (optional) number of inputs to be given special treatment
            these are always assumed to be the rightmost units
        """
        super().__init__(input_size, output_size)
        
        if context_size > 0:
            self.input_layer = torch.nn.Sequential(
                torch.nn.Linear(input_size, input_size - context_size),
                torch.nn.ReLU()
            )
        else:
            self.input_layer = torch.nn.Identity()
        # Architecture from https://github.com/riannevdberg/sylvester-flows/blob/master/models/VAE.py
        # TODO: generalise these parameters
        self.cnn = torch.nn.Sequential(
            GatedConvTranspose2d(input_size - context_size, 64, last_kernel_size, 1, 0),
            torch.nn.Dropout(dropout),
            GatedConvTranspose2d(64, 64, 5, 1, 2),
            torch.nn.Dropout(dropout),
            GatedConvTranspose2d(64, 32, 5, 2, 2, 1),
            torch.nn.Dropout(dropout),
            GatedConvTranspose2d(32, 32, 5, 1, 2),
            torch.nn.Dropout(dropout),
            GatedConvTranspose2d(32, 32, 5, 2, 2, 1),
            torch.nn.Dropout(dropout),
            GatedConvTranspose2d(32, input_channels, 5, 1, 2)
        )
        self.context_size = context_size
        self.output_layer = torch.nn.Conv2d(input_channels, output_channels, 1, 1, 0)
        
    def forward(self, inputs, **kwargs):
        h = self.input_layer(inputs)
        h = h.reshape(h.size(0), self.input_size - self.context_size, 1, 1)
        h = self.cnn(h)
        return self.output_layer(h).reshape(inputs.size(0), -1)    
    

