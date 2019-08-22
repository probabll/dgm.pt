import torch
import torch.nn.functional as F
from torch.distributions import Distribution
from dgm import parameterize_conditional
from dgm.nn import GatedConv2d, GatedConvTranspose2d
from dgm.nn import MADE

    
class Conditioner(torch.nn.Module):
    """
    A NN architecture maps a tensor of inputs to a tensor of outputs used to parameterize a distribution.
    """
    
    def __init__(self, input_size, output_size):
        super(Conditioner, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
    def forward(self, inputs):
        """
        :param inputs: [..., input_size]
        :return: [..., output_size]
        """
        raise NotImplementedError
        
        
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
       
    def forward(self, inputs) -> Distribution:
        """
        :param inputs: [..., input_size]
        :return: a parameterized Distribution
        """
        outputs = self.conditioner(inputs)
        return parameterize_conditional(self.dist_type, outputs, self.event_size)

    
class FFConditioner(Conditioner):
    """
    Conditions on inputs via a feed-forward transformation.
    """
    
    def __init__(self, input_size, output_size, context_size=0, hidden_sizes=[], hidden_activation=torch.nn.ReLU()):
        """
        :param input_size: number of units in the input 
        :param output_size: number of outputs         
        :param context_size: (optional) number of inputs to be given special treatment
            these are always assumed to be the rightmost units
        :param hidden_sizes: dimensionality of each hidden layer in the FFNN
        :param hidden_activation: what hidden activation to use
        """
        super(FFConditioner, self).__init__(input_size, output_size)
        self.hidden_activation = hidden_activation
        self.context_size = context_size
        # hidden layers
        net = [torch.nn.Linear(h0, h1) for h0, h1 in zip([input_size - context_size] + hidden_sizes, hidden_sizes)]        
        self.net = torch.nn.ModuleList(net)
        # output layer
        self.output_layer = torch.nn.Linear(hidden_sizes[-1], output_size)
        # context net
        if context_size > 0:
            ctxt_net = [torch.nn.Linear(context_size, units) for units in hidden_sizes]
            self.ctxt_net = torch.nn.ModuleList(ctxt_net)        
        else:
            self.ctxt_net = None
        
    def forward(self, inputs):
        if self.context_size > 0:  # some of the units are considered "context"       
            h, context = torch.split(inputs, self.input_size - self.context_size, -1)
            for t, c in zip(self.net, self.ctxt_net):
                h = self.hidden_activation(t(h) + c(context)) 
        else:
            h = inputs
            for t in self.net:
                h = self.hidden_activation(t(h))
        return self.output_layer(h)

    
class Conv2DConditioner(Conditioner):
    """
    Conditions on input shaped as [batch_size, width, height] using 2D convolutions
    """
    
    def __init__(self, input_size, output_size, context_size, width, height, 
                 output_channels, last_kernel_size):   
        """
        :param input_size: number of units in the input 
        :param output_size: number of outputs         
        :param context_size: (optional) number of inputs to be given special treatment
            these are always assumed to be the rightmost units
        :param width: for 2D convolutions we shape the inputs to [batch_size, width, height]
            (context units are not conditioned on via convolutions)
        :param height: for 2D convolutions we shape the inputs to [batch_size, width, height]
            (context units are not conditioned on via convolutions)
        :param output_channels: 
        :param last_kernel_size:
        """
        super(Conv2DConditioner, self).__init__(input_size, output_size)
        input_channels = (input_size - context_size) // (width * height)
        # Architecture from https://github.com/riannevdberg/sylvester-flows/blob/master/models/VAE.py
        # TODO: generalise these parameters
        self.cnn = torch.nn.Sequential(
            GatedConv2d(input_channels, 32, kernel_size=5, stride=1, padding=2),
            GatedConv2d(32, 32, kernel_size=5, stride=2, padding=2),
            GatedConv2d(32, 64, kernel_size=5, stride=1, padding=2),
            GatedConv2d(64, 64, kernel_size=5, stride=2, padding=2),
            GatedConv2d(64, 64, kernel_size=5, stride=1, padding=2),
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
    
    def __init__(self, input_size: int, output_size: int, context_size: int, 
                 input_channels: int, output_channels: int, last_kernel_size: int):
        """
        :param input_size: number of units in the input 
        :param output_size: number of outputs         
        :param context_size: (optional) number of inputs to be given special treatment
            these are always assumed to be the rightmost units
        :param input_channels: 
        :param output_channels:
        :param last_kernel_size:
        """
        super(TransposedConv2DConditioner, self).__init__(input_size, output_size)
        
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
            GatedConvTranspose2d(64, 64, 5, 1, 2),
            GatedConvTranspose2d(64, 32, 5, 2, 2, 1),
            GatedConvTranspose2d(32, 32, 5, 1, 2),
            GatedConvTranspose2d(32, 32, 5, 2, 2, 1),
            GatedConvTranspose2d(32, input_channels, 5, 1, 2)
        )
        self.context_size = context_size
        self.output_layer = torch.nn.Conv2d(input_channels, output_channels, 1, 1, 0)
        
    def forward(self, inputs):
        h = self.input_layer(inputs)
        h = h.reshape(h.size(0), self.input_size - self.context_size, 1, 1)
        h = self.cnn(h)
        return self.output_layer(h).reshape(inputs.size(0), -1)    
    

class MADEConditioner(Conditioner):
    """
    Wraps around MADE to have a nicer interface for a model.
    """
    
    def __init__(self, input_size: int, output_size: int, context_size: int,
                 hidden_sizes: list, hidden_activation=torch.nn.ReLU(), 
                 num_masks=1):
        """
        :param input_size: number of inputs to the conditioner
        :param output_size: number of outputs 
        :param context_size: number of (rightmost) input units assumed to be context 
            (context units are conditioned on without restriction)            
        :param hidden_sizes: dimensionality of each hidden layer in the MADE
        :param hidden_activation: activation for hidden layers in the MADE
        :param num_masks: use more than 1 to sample a number of random orderings
            1 implies the natural order        
        """
        super(MADEConditioner, self).__init__(input_size, output_size)
        self._made = MADE(
            nin=input_size - context_size,  # inputs we are autoregressive about
            nout=output_size,
            hidden_sizes=hidden_sizes,
            num_masks=num_masks,
            natural_ordering=num_masks==1,
            context_size=context_size,
            hidden_activation=hidden_activation
        )
        self.context_size = context_size
    
    def forward(self, inputs, num_samples=1, resample_mask=False):
        """
        :param x: data [..., input_dim]
        :param context: [..., context_size] or None
        :param num_samples: use more than 1 for ensembling (via average) outputs obtained with different masks
            (>1 implies resample_mask)
        :param resample_mask: whether or not to resample the masks
            (for example, due it every so often for order-agnostic training)
        :return: [..., output_dim]
        """
        if self.context_size > 0:            
            inputs, context = torch.split(inputs, self.input_size - self.context_size, -1)
        else:
            context = None            
        # [B, O]                
        outputs = self._made(inputs, context=context)
        if num_samples > 1:
            self._made.update_masks() 
            for s in range(num_samples - 1):                
                outputs += self._made(x, context=context)
                self._made.update_masks()                
            outputs = outputs / num_samples
        elif resample_mask:
            self._made.update_masks()
        return outputs    
