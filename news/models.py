# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RNN/LSTM/GCN models in Pytorch."""


import math

import torch
from torch import nn
from torch.nn.parameter import Parameter

if not __package__:
  import utils  # pylint: disable=g-import-not-at-top
else:
  from gnns_for_news import utils  # pylint: disable=g-bad-import-order,disable=g-import-not-at-top


class RNN(nn.Module):
  """RNN model."""

  def __init__(self, in_dim, hid_dim, num_label, num_layers=1, dropout=0.7):
    super(RNN, self).__init__()
    self.rnn = nn.RNN(input_size=in_dim, hidden_size=hid_dim,
                      num_layers=num_layers, dropout=dropout)
    self.linear = nn.Linear(hid_dim, num_label)

  def forward(self, x):
    _, hid_state = self.rnn(x)
    last_hidden_out = hid_state[-1].squeeze()
    return self.linear(last_hidden_out)


class LSTM(nn.Module):
  """LSTM model."""

  def __init__(self, in_dim, hid_dim, num_label, num_layers=2,
               dropout=0.7, bi_direct=False):
    super(LSTM, self).__init__()
    self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hid_dim,
                        num_layers=num_layers, bidirectional=bi_direct,
                        dropout=dropout, batch_first=True)
    self.linear = nn.Linear(hid_dim * 2 if bi_direct else hid_dim,
                            num_label)
    self.bi_direct = bi_direct
    self.num_layers = num_layers
    self.hid_dim = hid_dim

  def forward(self, x):
    _, (hid_state, cell_state) = self.lstm(x)
    if self.bi_direct:
      hid_state = hid_state.view(self.num_layers, 2, -1, self.hid_dim)
      # Concatenate forward, backward of last layer
      last_hidden_out = torch.cat([hid_state[-1][0], hid_state[-1][1]], 1)
    else:
      last_hidden_out = hid_state[-1].squeeze()

    return self.linear(last_hidden_out)


class GraphConvolution(nn.Module):
  """A graph convolution layer.

  It follows https://github.com/tkipf/pygcn and
  https://arxiv.org/abs/1609.02907.

  Attributes:
    in_dim: An integer indicating the dimension of input.
    out_dim: An integer indicating the dimension of output.
    weight: Torch tensor with dimension (in_dim, out_dim).
    bias: Torch tensor with dimension out_dim.
  """

  def __init__(self, in_dim, out_dim, bias=True):
    """Initializes the instance.

    Args:
      in_dim: An integer indicating the dimension of input.
      out_dim: An integer indicating the dimension of output.
      bias: An optional boolean variable indicating whether to have bias.
    """
    super().__init__()
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.weight = Parameter(torch.FloatTensor(in_dim, out_dim))
    self.bias = Parameter(torch.FloatTensor(out_dim)) if bias else None
    self.reset_parameters()

  def reset_parameters(self):
    """Initializes parameters following the default pytorch initialization."""
    stdv = 1. / math.sqrt(self.weight.size(1))
    self.weight.data.uniform_(-stdv, stdv)
    if self.bias is not None:
      self.bias.data.uniform_(-stdv, stdv)

  def forward(self, x_feature, adjacency_matrix):
    """Forward function.

    Args:
      x_feature: A tensor of input features.
      adjacency_matrix: A tensor of adjacency matrix.

    Returns:
      A tensor of output embedding.
    """
    support = torch.mm(x_feature, self.weight)
    output = torch.spmm(adjacency_matrix, support)
    if self.bias is not None:
      return output + self.bias
    else:
      return output


class GCN(nn.Module):
  """A 2-layer GCN model.

  It follows https://github.com/tkipf/pygcn and
  https://arxiv.org/abs/1609.02907.

  Attributes:
    gc1: A gcn layer.
    gc2: A gcn layer.
    dropout: A float indicating the dropout rate.
  """

  def __init__(self, in_dim, hid_dim, out_dim, dropout=0.0, bias=True):
    """Initializes the instance.

    Args:
      in_dim: An integer indicating the dimension of input.
      hid_dim: An integer indicating the dimension of hidden layer.
      out_dim: An integer indicating the dimension of output.
      dropout: An optional float variable indicating the dropout rate.
      bias: An optional variable indicating whether to include bias.
    """
    super().__init__()
    self.gc1 = GraphConvolution(in_dim, hid_dim, bias=bias)
    self.gc2 = GraphConvolution(hid_dim, out_dim, bias=bias)
    self.dropout = dropout

  def forward(self, x_feature, adjacency_matrix):
    """Forward function.

    This function applys two GCN layers with dropout.

    Args:
      x_feature: A tensor of input features of shape (number of nodes,
        feature dimension).
      adjacency_matrix: A tensor of adjacent matrix.

    Returns:
      A tensor of output embedding of shape (number of nodes, output dimension).
    """
    # replace relu by elu
    x_feature = nn.functional.elu(self.gc1(x_feature, adjacency_matrix))
    x_feature = nn.functional.dropout(x_feature, self.dropout,
                                      training=self.training)
    x_feature = self.gc2(x_feature, adjacency_matrix)
    return x_feature


class GCNRNN(nn.Module):
  """A GCN+RNN post sequence model.

  Attributes:
    gcn: A GCN class.
    rnn: A RNN or LSTM class.
  """

  def __init__(self, gcn_in_dim, gcn_hid_dim, gcn_out_dim, rnn_type,
               rnn_hid_dim, rnn_out_dim, rnn_num_layers=1, rnn_dropout=0.0,
               bi_direct=False, gcn_dropout=0.0, gcn_bias=True):
    """Initializes the instance.

    Args:
      gcn_in_dim: (integer) Dimension of GCN input.
      gcn_hid_dim: (integer) Dimension of GCN hidden layer.
      gcn_out_dim: (integer) Dimension of GCN output.
      rnn_type: (string, 'RNN' or 'LSTM') Whether to use RNN or LSTM.
      rnn_hid_dim: (integer) Dimension of RNN/LSTM hidden layer.
      rnn_out_dim: (integer) Dimension of RNN/LSTM output.
      rnn_num_layers: (integer, default=1) Number of layers in RNN/LSTM.
      rnn_dropout: (float in [0.0, 1.0], default=0.0) Dropout rate in RNN/LSTM.
      bi_direct: (boolean, default=False) Whether RNN/LSTM is bidirectional.
      gcn_dropout: (float in [0.0, 1.0], default=0.0) Dropout rate in GCN.
      gcn_bias: (boolean, default=True) Whether to include bias in GCN.
    """
    super().__init__()
    self.gcn = GCN(in_dim=gcn_in_dim, hid_dim=gcn_hid_dim, out_dim=gcn_out_dim,
                   dropout=gcn_dropout, bias=gcn_bias)
    if rnn_type == 'RNN':
      self.rnn = RNN(in_dim=gcn_out_dim, hid_dim=rnn_hid_dim,
                     num_label=rnn_out_dim, num_layers=rnn_num_layers,
                     dropout=rnn_dropout)
    elif rnn_type == 'LSTM':
      self.rnn = LSTM(in_dim=gcn_out_dim, hid_dim=rnn_hid_dim,
                      num_label=rnn_out_dim, num_layers=rnn_num_layers,
                      dropout=rnn_dropout, bi_direct=bi_direct)

  def forward(self, batch_x):
    """Forward function.

    Args:
      batch_x: A batch of input data.

    Returns:
      A batch tensor of output embedding.
    """
    batch_x_post_embedding = []
    for x_sequence in batch_x:
      post_embeddings = torch.stack(
          [self.gcn(x_feature, adj)[0] for x_feature, adj in x_sequence])
      batch_x_post_embedding.append(post_embeddings)
    batch_x_post_embedding = utils.data_padding(batch_x_post_embedding)
    output = self.rnn(batch_x_post_embedding)
    return output

