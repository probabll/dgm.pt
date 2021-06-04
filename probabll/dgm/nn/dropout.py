import torch
from torch import nn


class WordDropout(nn.Module):
    """
    Word Dropout, as described in [1].
    Replace a word with UNK token with probability p, padding tokens are excluded.

    [1] Bowman, S. R., Vilnis, L., Vinyals, O., Dai, A. M., Jozefowicz, R., & Bengio, S. (2015).
        Generating sentences from a continuous space.
    """

    def __init__(self, p, unk_idx: int, padding_idx: int = 0):
        super(WordDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("p should be a probability, 0 < p < 1.")
        self.p = p
        self.unk_idx = unk_idx
        self.padding_idx = padding_idx

    def forward(self, x):
        if self.p == 0 or not self.training:
            return x

        # Replace unpadded words with unk with probability self.p
        p_mask = torch.rand(*x.size()) < self.p
        unk = torch.full_like(x, self.unk_idx).to(x.device)
        # mask = not padding & probs < p_dropout
        unk_mask = (~torch.eq(x, self.padding_idx)) * p_mask.to(x.device)
        return torch.where(unk_mask, unk, x)
