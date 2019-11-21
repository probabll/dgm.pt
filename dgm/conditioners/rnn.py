"""
Conditioners based on RNNs
"""
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .conditioner import Conditioner
from dgm.nn import WordDropout


class RNNConditioner(Conditioner):
    """
    Many-to-one RNN conditioner, where the output is the last output of the RNN.
    """

    def __init__(self, embedding_size, hidden_size, output_size, num_layers=1,
                 rnn_type=torch.nn.LSTM, bidirectional=True, embedding=None,
                 vocab_size=None, padding_idx=0):
        """
        :param embedding_size: Size of embeddings, not inferred if embedding is given.
        :param hidden_size: Size of RNN layers
        :param output_size: Size of final linear layer
        :param num_layers: (optional) number of layers in RNN, defaults to 1
        :param rnn_type: (optional) Type of RNN used, typically nn.LSTM or nn.GRU.
            defaults to nn.LSTM
        :param bidirectional: (optional) use bidirectional RNN, defaults to True
        :param embedding: (optional) nn.Embedding layer,
            use if encoder and decoder share embedidngs. defaults to None
        :param vocab_size: (optional) If no embedding is given,
            new embedding is constructed with vocab_size. defaults to None
        :param padding_idx: (optional) Idx of pad token, defaults to 0
        :raises ValueError: Either an embedding should be given,
            or a vocab size to construct a new embedding.
        """

        # RNN has variable input size
        super().__init__(None, output_size)

        if embedding:
            self.embedding = embedding
        elif vocab_size:
            self.embedding = torch.nn.Embedding(vocab_size, embedding_size,
                                                padding_idx=padding_idx)
        else:
            raise ValueError("Either an embedding should be given, or a vocab size to construct a new embedding.")

        self.rnn = rnn_type(embedding_size, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional)

        rnn_out_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc_out = torch.nn.Linear(rnn_out_size, output_size)

    def forward(self, x, lengths=None, **kwargs):
        # x [B, L], lengths [B]
        out = self.embedding(x)

        if isinstance(lengths, torch.LongTensor):
            out = pack_padded_sequence(
                out, lengths, batch_first=True, enforce_sorted=False)

        out, _ = self.rnn(out)

        if isinstance(lengths, torch.LongTensor):
            out, _ = pad_packed_sequence(out, batch_first=True)

        # select last output from RNN
        if lengths is None:
            lengths = torch.LongTensor(
                [out.size(1)] * out.size(0)
            ).to(out.device)
        out = out[torch.arange(out.size(0), dtype=torch.long), lengths - 1]
        return self.fc_out(out)


class RNNLMConditioner(Conditioner):
    """
    A one-to-many RNN conditioner, with a few additions to reduce posterior collapse.

    To make this compatible with the AutoregressiveLikelihood container:

    conditioner = RNNLMConditioner(...)
    likelihood = AutoregressiveLikelihood(None, Categorical, conditioner)
    [...]
    z = p_z.rsample()
    history = x_batch[..., :-1]
    p_x = likelihood(None, history, z=z)
    
    labels = x_batch[..., 1:]
    logits = p_x.logits.transpose(-2, -1)
    ll = -F.cross_entropy(logits, labels, ignore_index=PAD_IDX, 
                          reduction='none').sum(-1)
    """

    def __init__(self, embedding_size, z_size, hidden_size, output_size, num_layers=1,
                 rnn_type=torch.nn.LSTM, embedding=None, vocab_size=None, p_embedding_dropout=0., 
                 p_word_dropout=0., unk_idx=None, padding_idx=0, z_every_step=False):
        """
        :param embedding_size: Size of embedding.
        :param z_size: Size of latent vector RNN is conditioned on.
        :param hidden_size: Size of RNN layers.
        :param output_size: Size of single output.
        :param num_layers: (optional) Number of RNN layers, defaults to 1.
        :param rnn_type: (optional) Type of RNN used, typically nn.LSTM or nn.GRU,
            defaults to nn.LSTM.
        :param embedding: (optional) nn.Embedding layer,
            use if encoder and decoder share embedidngs. defaults to None.
        :param vocab_size: (optional) If no embedding is given,
            new embedding is constructed with vocab_size. defaults to None.
        :param p_word_dropout: (optional) Word dropout chance, defaults to 0.
        :param unk_idx: idx of UNK token, defaults to 3.
        :param padding_idx: idx of PAD token, defaults to 0.
        :param z_every_step: If True, z is concatenated to the word embedding at every timestep,
            defaults to False.
        """
        # NOTE: output_size is really [B, sequence_length, output_size],
        # where sequence_length is variable length.
        super().__init__(z_size, output_size)

        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.p_word_dropout = p_word_dropout
        self.z_every_step = z_every_step

        if embedding:
            self.embedding = embedding
        elif vocab_size:
            self.embedding = torch.nn.Embedding(vocab_size, embedding_size,
                                                padding_idx=padding_idx)
        else:
            raise ValueError("Either an embedding should be given, \
                              or a vocab size to construct a new embedding.")

        self.dropout_emb = torch.nn.Dropout(p=p_embedding_dropout)

        if self.p_word_dropout:
            if unk_idx is None:
                raise ValueError("When using word dropout, unk_idx should be given.")
            self.word_dropout = WordDropout(self.p_word_dropout, unk_idx, padding_idx)

        if z_every_step:
            # [B, L, Z+E] -> [B, L, E]
            self.fc_in = torch.nn.Linear(embedding_size + z_size, embedding_size)

        if self.rnn_type == torch.nn.LSTM:
            self.fc_h0 = torch.nn.Linear(z_size, hidden_size * 2)
        else:
            self.fc_h0 = torch.nn.Linear(z_size, hidden_size)

        self.rnn = rnn_type(input_size=embedding_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc_out = torch.nn.Linear(hidden_size, output_size)

    def _forward(self, x, z, h0=None):
        """
        Use for both forward pass and sampling/training without teacher forcing.
        Construct initial hidden state h0 from z if no h0 is given.
        """
        # x = [B, L], z = [B, Z]
        # Initial hidden state from z
        if h0 is None:
            h0 = torch.tanh(self.fc_h0(z))
            h0 = h0.expand((self.num_layers, *h0.shape))
            if self.rnn_type == torch.nn.LSTM:
                # If lstm, split to hidden and cell state
                h0 = torch.chunk(h0, 2, dim=-1)
                h0 = tuple([t.contiguous() for t in h0])

        # Embedding
        if self.p_word_dropout:
            x = self.word_dropout(x)
        x_emb = self.dropout_emb(self.embedding(x))

        # Concatenate z every timestep
        if self.z_every_step:
            z_exp = z.unsqueeze(1).expand((-1, x_emb.size(1), -1))
            x_emb = self.fc_in(torch.cat([x_emb, z_exp], dim=-1))

        # RNN
        out, h_out = self.rnn(x_emb, h0)
        out = self.fc_out(out)
        return out, h_out

    def forward(self, x, z=None):
        return self._forward(x, z)[0]

    def sample(self, x, z=None, length=60, temperature=1):
        outcome = torch.zeros(x.size(0), length).long()
        outcome[:, 0] = x.squeeze(-1)

        with torch.no_grad():
            h0 = None
            # For step in length, generate new sample with previous x and
            # hidden state
            for i in range(1, length):
                out, h0 = self._forward(x, z, h0=h0)
                out = torch.softmax(out / temperature, dim=-1).squeeze(1)
                x = torch.multinomial(out, 1)
                outcome[:, i] = x.squeeze(-1).to(outcome.device)

        return outcome

