
from torch.autograd import Variable
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LanguageModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_size,
                 num_layers):
        """Initializes the language model.
        Args:
            vocab_size: Number of words in the vocabulary.
            embedding_dim: Dimension of the word embeddings.
            hidden_size: Number of units in each LSTM hidden layer.
            num_layers: Number of hidden layers.
        """
        # Always do this !!!
        super(LanguageModel, self).__init__()

        # Store parameters
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define layers
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim,
                                            padding_idx=0)
        self.rnn = nn.GRU(embedding_dim, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.activation = nn.LogSoftmax(dim=2)

    def forward(self, x, lengths=None, hidden=None):
        """Computes a forward pass of the language model.
        Args:
            x: A LongTensor w/ dimension [seq_len, batch_size].
            lengths: The lengths of the sequences in x.
            hidden: Hidden state to be fed into the lstm.
        Returns:
            net: Probability of the next word in the sequence.
            hidden: Hidden state of the lstm.
        """
        seq_len, batch_size = x.size()
        # If no hidden state is provided, then default to zeros.
        if hidden is None:
            hidden = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
            if torch.cuda.is_available():
                hidden = hidden.cuda()

        net = self.word_embeddings(x)
        if lengths is not None:
            lengths = lengths.data.view(-1).tolist()
            net = pack_padded_sequence(net, lengths)
        net, hidden = self.rnn(net, hidden)
        if lengths is not None:
            net, _ = pad_packed_sequence(net)
        net = self.fc(net)
        net = self.activation(net)

        return net, hidden