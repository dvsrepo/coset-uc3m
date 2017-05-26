
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable

class RNNClassifier(nn.Module):

    def __init__(self, input_dim, embedding_size, rnn_hidden_size, output_dim, num_layers=1, dropout=0.1):
        super(RNNClassifier, self).__init__()
        self.embed = nn.Embedding(input_dim, embedding_size)
        self.rnn = nn.GRU(input_size=embedding_size, hidden_size=rnn_hidden_size,
                          dropout=dropout, num_layers=num_layers, bidirectional=True)

        self.out = nn.Linear(rnn_hidden_size*2, output_dim)
        self.init_weights()
        self.init_bias()


    def init_weights(self):
        initrange = 0.1
        for weight in self.rnn.parameters():
            weight.data.uniform_(-initrange, initrange)


    def init_bias(self):
        for names in self.rnn._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.rnn, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)


    def forward(self, input):
        input = self.embed(input)
        input = F.dropout(input, training=self.training, p=0.2)
        out, ht = self.rnn(input) #(h0, c0)
        out = self.out(out[-1].view(-1, out.size(2)))
        return out
