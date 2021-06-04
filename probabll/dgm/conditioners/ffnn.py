"""
Conditioners based on FFNNs
"""

import torch

from probabll.dgm.nn import MADE

from .conditioner import Conditioner


class FFConditioner(Conditioner):
    """
    Conditions on inputs via a feed-forward transformation.
    """

    def __init__(
        self,
        input_size,
        output_size,
        context_size=0,
        hidden_sizes=[],
        hidden_activation=torch.nn.ReLU(),
    ):
        """
        :param input_size: number of units in the input
        :param output_size: number of outputs
        :param context_size: (optional) number of inputs to be given special treatment
            these are always assumed to be the rightmost units
        :param hidden_sizes: dimensionality of each hidden layer in the FFNN
        :param hidden_activation: what hidden activation to use
        """
        super().__init__(input_size, output_size)
        self.hidden_activation = hidden_activation
        self.context_size = context_size
        # hidden layers
        net = [
            torch.nn.Linear(h0, h1)
            for h0, h1 in zip([input_size - context_size] + hidden_sizes, hidden_sizes)
        ]
        self.net = torch.nn.ModuleList(net)
        # output layer
        self.output_layer = torch.nn.Linear(hidden_sizes[-1], output_size)
        # context net
        if context_size > 0:
            ctxt_net = [torch.nn.Linear(context_size, units) for units in hidden_sizes]
            self.ctxt_net = torch.nn.ModuleList(ctxt_net)
        else:
            self.ctxt_net = None

    def forward(self, inputs, **kwargs):
        if self.context_size > 0:  # some of the units are considered "context"
            h, context = torch.split(inputs, self.input_size - self.context_size, -1)
            for t, c in zip(self.net, self.ctxt_net):
                h = self.hidden_activation(t(h) + c(context))
        else:
            h = inputs
            for t in self.net:
                h = self.hidden_activation(t(h))
        return self.output_layer(h)


class MADEConditioner(Conditioner):
    """
    Wraps around MADE to have a nicer interface for a model.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list,
        hidden_activation=torch.nn.ReLU(),
        num_masks=1,
        context_size=0,
    ):
        """
        :param input_size: number of inputs to the conditioner
        :param output_size: number of outputs
        :param hidden_sizes: dimensionality of each hidden layer in the MADE
        :param hidden_activation: activation for hidden layers in the MADE
        :param num_masks: use more than 1 to sample a number of random orderings
            1 implies the natural order
        :param context_size: number of (rightmost) input units assumed to be context
            (context units are conditioned on without restriction)
        """
        super().__init__(input_size, output_size)
        self._made = MADE(
            nin=input_size - context_size,  # inputs we are autoregressive about
            nout=output_size,
            hidden_sizes=hidden_sizes,
            num_masks=num_masks,
            natural_ordering=num_masks == 1,
            context_size=context_size,
            hidden_activation=hidden_activation,
        )
        self.context_size = context_size

    def forward(self, inputs, num_samples=1, resample_mask=False, **kwargs):
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
            inputs, context = torch.split(
                inputs, [self.input_size - self.context_size, self.context_size], -1
            )
        else:
            context = None
        # [B, O]
        outputs = self._made(inputs, context=context)
        if num_samples > 1:
            self._made.update_masks()
            for s in range(num_samples - 1):
                outputs += self._made(inputs, context=context)
                self._made.update_masks()
            outputs = outputs / num_samples
        elif resample_mask:
            self._made.update_masks()
        return outputs
