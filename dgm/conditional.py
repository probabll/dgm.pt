import torch
import torch.nn.functional as F
from torch.distributions import Distribution
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from dgm import parameterize_conditional
from dgm.nn import GatedConv2d, GatedConvTranspose2d
from dgm.nn import MADE, WordDropout


class Conditioner(torch.nn.Module):
    """
    A NN architecture maps a tensor of inputs to a tensor of outputs used to parameterize a distribution.
    """
    
    def __init__(self, input_size, output_size):
        super(Conditioner, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
    def forward(self, inputs, **kwargs):
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
       
    def forward(self, inputs, **kwargs) -> Distribution:
        """
        :param inputs: [..., input_size]
        :return: a parameterized Distribution
        """
        outputs = self.conditioner(inputs, **kwargs)
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

    
class Conv2DConditioner(Conditioner):
    """
    Conditions on input shaped as [batch_size, width, height] using 2D convolutions
    """
    
    def __init__(self, input_size, output_size, width, height, 
                 output_channels, last_kernel_size, context_size=0):   
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
    
    def __init__(self, input_size: int, output_size: int, 
                 input_channels: int, output_channels: int, last_kernel_size: int, context_size=0):
        """
        :param input_size: number of units in the input 
        :param output_size: number of outputs         
        :param input_channels: 
        :param output_channels:
        :param last_kernel_size:
        :param context_size: (optional) number of inputs to be given special treatment
            these are always assumed to be the rightmost units
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
        
    def forward(self, inputs, **kwargs):
        h = self.input_layer(inputs)
        h = h.reshape(h.size(0), self.input_size - self.context_size, 1, 1)
        h = self.cnn(h)
        return self.output_layer(h).reshape(inputs.size(0), -1)    
    

class MADEConditioner(Conditioner):
    """
    Wraps around MADE to have a nicer interface for a model.
    """
    
    def __init__(self, input_size: int, output_size: int, 
                 hidden_sizes: list, hidden_activation=torch.nn.ReLU(), 
                 num_masks=1, context_size=0):
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
            inputs, context = torch.split(inputs, self.input_size - self.context_size, -1)
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
        super(RNNConditioner, self).__init__(None, output_size)

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
        super(RNNLMConditioner, self).__init__(z_size, output_size)

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
            print('H0')
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
