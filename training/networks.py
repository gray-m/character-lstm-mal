import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F


def save_model(network, optimizer, save_path):
    checkpoint = {
            'net_params': {
                'vocab_size': network.vocab_size,
                'lstm_n_hidden': network.lstm_n_hidden,
                'lstm_n_layers': network.lstm_n_layers,
            },
            'net_state': network.state_dict(),
            'opt_state': optimizer.state_dict(),
    }

    with open(save_path, 'wb') as f:
        pickle.dump(checkpoint, f)


def one_hot_converter(vocab):
    onehot = nn.Embedding(len(vocab)+1, len(vocab)+1)
    onehot.weight.data = torch.eye(len(vocab)+1)
    for param in onehot.parameters():
        param.requires_grad = False
    return onehot


class CharPredictor(nn.Module):


    def __init__(self,
                 vocab_size,
                 lstm_n_hidden,
                 lstm_n_layers,
                 lstm_dropout=0.):
        super(CharPredictor, self).__init__()
        self.vocab_size = vocab_size
        self.lstm_n_hidden = lstm_n_hidden
        self.lstm_n_layers = lstm_n_layers
        self.lstm_layer = nn.LSTM(
                            input_size=vocab_size+1,
                            hidden_size=lstm_n_hidden,
                            num_layers=lstm_n_layers,
                            batch_first=True,
                            dropout=lstm_dropout,
                          )
        self.decoder = nn.Linear(lstm_n_hidden, vocab_size+1)
        self.logsoft = nn.LogSoftmax(dim=2)
        self.h = None
        self.c = None


    def init_hiddens(self, batch_size):
        self.h = torch.zeros(self.lstm_n_layers, batch_size, self.lstm_n_hidden)
        self.c = torch.zeros(self.lstm_n_layers, batch_size, self.lstm_n_hidden)


    def forward(self, inpt, init_hiddens=True):
        if init_hiddens:
            self.init_hiddens(len(inpt))
        lstm_out, (self.h, self.c) = self.lstm_layer(inpt, (self.h,self.c))
        out_raw = self.decoder(lstm_out)
        out = self.logsoft(out_raw)
        return out


    @classmethod
    def from_checkpoint(cls, checkpoint, dropout=0.):
        inst = cls(**checkpoint['net_params'], lstm_dropout=dropout)
        inst.load_state_dict(checkpoint['net_state'])
        return inst

