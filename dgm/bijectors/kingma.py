import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from .bijection import Bijection


EPS=1e-4


class AutoregressiveLinear(nn.Module):
    """
    This layer realises linear transformations
        y = f(c, x)
    autoregressive in x, i.e. \pdv{y_i}{x_j} = 0 for i >= j (or possibly i > j, see arguments in __init__), where
        x and y have the same dimensionality
        c is a context vector.
    The context vector is optional.
    """

    def __init__(self, output_size, context_size=0, diagonal_zeros=True, bias=True):
        """
        Input size is output_size + context_size
        :param output_size: number D of output units
        :param context_size: possibly 0 number of context units
        :param diagonal_zeros: True means \pdv{y_i}{x_j} = 0 for i >= j, False means \pdv{y_i}{x_j} = 0 for i > j.
        :param bias:
        """
        super().__init__()

        self.context_size = context_size
        self.output_size = output_size
        self.input_size = context_size + output_size
        self.diagonal_zeros = diagonal_zeros
        self._d = 1 if diagonal_zeros else 0  # number of diagonals to exclude (starting from the main)

        self.weight = nn.parameter.Parameter(
            nn.init.kaiming_normal_(
                torch.Tensor(self.output_size, self.input_size)
            )
        )
        self.bias = torch.nn.Parameter(
            torch.nn.init.uniform_(
                torch.Tensor(output_size),
                -1 / math.sqrt(output_size),
                1 / math.sqrt(output_size)
            )
        ) if bias else 0

    def forward(self, inputs, context=None):
        """
        :param inputs: [..., D]
        :param context: [..., H]
        :return: [..., D]
        """
        if self.context_size == 0:
            return F.linear(inputs, self.weight.tril(-self._d), self.bias)
        else:  # outputs depend freely on the context units
            return F.linear(torch.cat([context, inputs], dim=-1), self.weight.tril(self.context_size - self._d), self.bias)


class KingmaGating(Bijection):
    """
    Autoregressive linear transformation
        y = u(x) (1 - s(x)) + s(x) * x
    where u(x) and s(x) are AutoregressiveLinear (with masked diagonals).
    * u(x) is linear activated
    * s(x) is sigmoid-activated
    Thus outputs are in R^D.
    """

    def __init__(self, units, context_size, hidden_activation=nn.ELU,
            dropout=0., forget_bias=0., flip=False):
        super().__init__()
        self.units = units
        self.context_size = context_size
        self.dropout_layer = nn.Dropout(p=dropout)
        self.flip = flip
        self.flip_ids = torch.arange(units - 1, -1, -1).long()

        if context_size > 0:
            self.bridge = nn.Sequential(
                nn.Dropout(dropout),
                AutoregressiveLinear(units, context_size=context_size, diagonal_zeros=False, bias=True),
                hidden_activation(),
                nn.Dropout(dropout),
                AutoregressiveLinear(units, diagonal_zeros=False, bias=True),
                hidden_activation(),
            )
        else:
            self.bridge = nn.Sequential(
                nn.Dropout(dropout),
                AutoregressiveLinear(units, diagonal_zeros=False, bias=True),
                hidden_activation(),
                nn.Dropout(dropout),
                AutoregressiveLinear(units, diagonal_zeros=False, bias=True),
                hidden_activation(),
            )
        self.loc_layer = nn.Sequential(
            nn.Dropout(dropout),
            AutoregressiveLinear(units, diagonal_zeros=True)
        )
        self.gate_layer = nn.Sequential(
            nn.Dropout(dropout),
            AutoregressiveLinear(units, diagonal_zeros=True)
        )
        self.forget_bias = forget_bias

    def forward(self, inputs, context=None):
        # this part conditions on x_{<=d}
        # [..., D]
        if self.flip:
            inputs = inputs[:, self.flip_ids]  # TODO: double check this
        if context is None:
            h = self.bridge(inputs)
        else:
            h = self.bridge(torch.cat([context, inputs], dim=-1))
        # this part conditions on x_{<d}
        # [..., D]
        loc = self.loc_layer(h)
        gate = torch.sigmoid(self.gate_layer(h) + self.forget_bias)
        return gate * inputs + (1 - gate) * loc, torch.log(gate + EPS)

    def inverse(self, outputs, context=None):
        # [..., D]
        inputs = torch.zeros_like(outputs)
        log_det_jac = 0.
        for d in range(1, self.units + 1):
            # this part conditions on x_{<=d}
            # [..., D]
            if context is None:
                h = self.bridge(inputs)
            else:
                h = self.bridge(torch.cat([context, inputs], dim=-1))
            # this part conditions on x_{<d}
            # [..., D]
            loc = self.loc_layer(h)
            gate = torch.sigmoid(self.gate_layer(h) + self.forget_bias)
            inputs = ((outputs - (1 - gate) * loc) / gate).tril(d - 1)
            log_det_jac = - torch.log(gate + EPS)

        if self.flip:  # TODO: double check this
            inputs = inputs[:, self.flip_ids]
        return inputs, log_det_jac

