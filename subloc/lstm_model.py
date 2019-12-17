import torch
import torch.nn as nn

from torch.autograd import Variable 
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import softmax

class SeqClassifier(nn.Module):
  def __init__(self, device, d_embed=32, d_project=256, d_rnn=128, n_layers=2,
                 bi_dir=True, n_classes=10, dropout=0.5, rnn_dropout=0.3):
    super(SeqClassifier, self).__init__()

    self.device = device
    self.n_layers = n_layers
    self.directions= 2 if bi_dir else 1
    self.d_project = d_project
    self.d_rnn = d_rnn
    rnn_dropout = rnn_dropout if n_layers > 1 else 0

    self.dropout = nn.Dropout(dropout)

    # Linear projection layer
    self.linear_in = nn.Linear(d_embed, d_project)
    
    # LSTM (optinally bi-directional)
    self.LSTM = nn.LSTM(d_project, hidden_size=d_rnn, num_layers=n_layers,
                            batch_first=True, dropout=rnn_dropout, bidirectional=bi_dir)

    # Logit layer
    self.linear_out = nn.Linear(self.directions*d_rnn, n_classes, bias=True)

  def init_hidden(self, batch_size):
    h0 = torch.randn(self.n_layers*self.directions, batch_size, self.d_rnn).to(self.device)
    c0 = torch.randn(self.n_layers*self.directions, batch_size, self.d_rnn).to(self.device)
    
    h0 = Variable(h0)
    c0 = Variable(c0)

    return (h0, c0)

  def forward(self, batch, seq_lengths):
    
    batch_size, max_seq_len, d_embed = batch.shape

    x = batch.view(batch_size*max_seq_len, d_embed)
    x = self.dropout(x)
    projected = self.linear_in(x)
    projected = projected.view(batch_size, max_seq_len, self.d_project)

    # Pack sequence
    packed = pack_padded_sequence(projected, seq_lengths, batch_first=True, enforce_sorted=True)

    # Initialize lstm hidden and cell state
    initial_state_params = self.init_hidden(batch_size)
    rnn_out, (h_n, c_n) = self.LSTM(packed, initial_state_params)

    # Unpack sequence
    h, _ = pad_packed_sequence(rnn_out, batch_first=True, padding_value=0)

    # Select the last hidden state for last timestep
    idx = seq_lengths-1
    batch_idx = torch.arange(batch_size)
    h_last = torch.stack([h[b,i,:] for b,i in zip(batch_idx, idx)], dim=0)
    
    x = self.dropout(h_last)
    logits = self.linear_out(x)

    return softmax(logits, dim=1)