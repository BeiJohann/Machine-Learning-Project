import torch
import torch.nn as nn

from argparse import Namespace


values = Namespace(
    seq_size=16,
    batch_size=16,
    embedding_size=128,
    lstm_size=128,
    gradients_norm=5,
)


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, lstm_size, num_layers=2, dropout=0.1):
        super(RNN, self).__init__()
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, lstm_size, num_layers, dropout=dropout, batch_first=True)
        self.lin1 = nn.Linear(lstm_size, vocab_size)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        output = self.lin1(output)
        return output, state

    def zero_state(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.lstm_size)
