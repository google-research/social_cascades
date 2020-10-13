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

"""Util functions for data processing and embedding generation.

It contains the util functions for graph loading, node2vec embedding
generation, pre-trained bert embedding generation, data normalization, graph
filtering, data padding, data balancing and train/test data split.

"""
import collections
import os

from absl import logging
from gensim.models import KeyedVectors
import networkx as nx
from node2vec import Node2Vec  # See README for installing Node2Vec package.
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import torch
from torch.nn.utils import rnn
from transformers import BertModel  # See README for installing transformers.
from transformers import BertTokenizer


def load_graph(filename, load_method='csv', directed=True):
  """Loads graph.

  Args:
    filename: A string of graph file name.
    load_method: A string denoting whether the graph file is csv or gpickle.
    directed: An optional boolean variable, which decides whether the returned
      graph is directed or undirected.

  Returns:
    A graph object.

  Raises:
    ValueError: If the value for the parameter is not correct.
  """
  if not filename:
    raise ValueError('The parameter filename should not be None.')
  logging.info('Loading Graph...')
  if load_method == 'gpickle':
    graph = nx.read_gpickle(filename)
  elif load_method == 'csv':
    graph = nx.DiGraph()
    with open(filename, 'r') as f:
      for i, line in enumerate(f):
        if i > 0:
          edge = line.strip().split(',')
          graph.add_edge(edge[0], edge[1], weight=float(edge[2]))
  else:
    raise ValueError('The value of load_method is not supported.')
  if directed:
    graph = graph.to_directed()
  else:
    graph = graph.to_undirected()
  logging.info('Graph information %s', nx.info(graph))
  return graph


def get_n2v_graph_embedding(filename, graph_gen=True, graph=None,
                            dimensions=128, walk_length=30, num_walks=100,
                            workers=4, win_size=10, normalize_type=None):
  """Generates/Loads graph embedding with node2vec.

  Args:
    filename: A string of graph embedding file name.
    graph_gen: A boolean variable denoting whether generate graph embedding.
    graph: A graph object.
    dimensions: An integer denoting the dimension for graph embedding.
    walk_length: An integer denoting the walk length.
    num_walks: An integer denoting the number of walks.
    workers: An integer denoting the number of workers.
    win_size: An integer denoting the window size.
    normalize_type: An optional variable, which decides whether to normalize
      the graph embedding. If it is None, there is no normalization. Available
      normalization options are minmax and standard.

  Returns:
    embeddings_dict: A dictionary, where keys are node names and values are
      their graph embedding.

  Raises:
    ValueError: If the value for the parameter is not correct.
  """
  if not filename:
    raise ValueError('The parameter filename should not be None.')
  if graph_gen:
    if graph is None:
      raise ValueError('The parameter graph should not be None.')
    node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=walk_length,
                        num_walks=num_walks, workers=workers)
    n2v_model = node2vec.fit(window=win_size, min_count=1, batch_words=4)
    n2v_model.wv.save_word2vec_format(filename)
    logging.info('Saved graph embedding.')
  embeddings_kv = KeyedVectors.load_word2vec_format(filename)

  embeddings_dict = {}
  for node in embeddings_kv.vocab:
    embeddings_dict[node] = embeddings_kv[node]
  if normalize_type:
    embeddings_dict = normalize_embeddings(embeddings_dict, normalize_type)
  return embeddings_dict


def graph_filter_with_degree(graph, low, high, author_set):
  """Removes self-loop edges, and nodes with low/high degree.

  Args:
    graph: A graph object.
    low: An integer denoting the low degree threshold.
    high: An integer denoting the high degree threshold.
    author_set: A set of authors of the selected posts, and they should be kept
      in the graph.

  Returns:
    A graph object.

  Raises:
    ValueError: If the value for the parameter is not correct.
  """
  if graph is None:
    raise ValueError('The parameter graph should not be None.')
  graph.remove_edges_from(nx.selfloop_edges(graph))
  remove = [node for node, degree in dict(graph.degree()).items()
            if (degree < low or degree > high) and node not in author_set]
  graph.remove_nodes_from(remove)
  return graph


def load_embeddings(id_sequence, embeddings_dict):
  """Turns user list into sequence of user graph embedding.

  Args:
    id_sequence: A list of user ids(same as user names).
    embeddings_dict: A dictionary, where keys are user ids and values are
      their graph embedding.

  Returns:
    embeddings: A numpy.array of the sequence features, and its shape is
    (sequence length, graph embedding size).
    id_list: A list of ids which have embeddings in the id_sequence.
  """
  embeddings, id_list = [], []
  for node in id_sequence:
    if node in embeddings_dict:
      embeddings.append(embeddings_dict[node])
      id_list.append(node)
  embeddings = None if not id_list else np.array(embeddings)
  return embeddings, id_list


def normalize_embeddings(embeddings_dict, normal_type='minmax'):
  """Normalizes graph embedding.

  Args:
    embeddings_dict: A dictionary, where keys are user ids and values are
      their graph embedding.
    normal_type: An optional variable, which decides the normalization type
      minmax or standard.

  Returns:
    A dictionary, where keys are user ids and values are their normalized
      graph embedding.
  """
  if normal_type == 'minmax':
    scaler = MinMaxScaler()
  elif normal_type == 'standard':
    scaler = StandardScaler()
  emb_array = np.array(list(embeddings_dict.values()))
  scaler.fit(emb_array)
  embeddings_dict_scaled = {}
  for node in embeddings_dict:
    embeddings_dict_scaled[node] = scaler.transform(
        embeddings_dict[node].reshape(1, -1))[0]
  return embeddings_dict_scaled


def load_input_with_label(sequence_df, embeddings_dict, task):
  """Loads sequence dataset with graph embedding.

  Args:
    sequence_df: A pandas.dataframe containing the sequence dataset.
    embeddings_dict: A dictionary, where keys are user ids and values are
      their graph embedding.
    task: A string of task name.

  Returns:
    x_sequence: A list of sequence features, and each element inside the list is
      a numpy.array of shape (sequence length, graph embedding size).
    y_label: A numpy.array of indexes in label_list for each sequence label.
    label_list: A list of unique labels.

  Raises:
    ValueError: If the value for the parameter is not correct.
  """
  if sequence_df is None:
    raise ValueError('The parameter sequence_df should not be None.')
  if embeddings_dict is None:
    raise ValueError('The parameter embeddings_dict should not be None.')
  sequence_df['user_emb'] = sequence_df.apply(
      lambda x: load_embeddings(x['user_list'], embeddings_dict)[0], axis=1)
  sequence_df = sequence_df.dropna()
  x_sequence = sequence_df['user_emb'].tolist()
  if task == 'sr':
    y_label, label_list = pd.factorize(sequence_df['sr'])
  elif task in ['cat', 'fake']:
    y_label, label_list = pd.factorize(sequence_df['label'])
  else:
    raise ValueError('The value of task is not supported.')
  return x_sequence, y_label, label_list


def split_data_idx(num_idx, train_ratio, val_ratio, shuffle=True):
  """Generates train, test index.

  Args:
    num_idx: An integer denoting the number of dataset.
    train_ratio: A float number denoting the ratio for training dataset.
    val_ratio: A float number denoting the ratio for validation dataset.
    shuffle: An optional variable, which decides whether to shuffle the dataset.

  Returns:
    train_index: An one-dimension numpy.array of training dataset indexes.
    test_index: An one-dimension numpy.array of testing dataset indexes.
  """
  split1 = int(train_ratio * num_idx)
  split2 = int((train_ratio + val_ratio) * num_idx)
  index = np.arange(num_idx)
  if shuffle:
    np.random.shuffle(index)
  train_index = index[:split1]
  val_index = index[split1: split2]
  test_index = index[split2:]
  return train_index, val_index, test_index


def balance_data(x_sequence, y_label, class_size=np.inf, shuffle=True):
  """Balances binary/multi-class dataset.

  Uses downsampling to let different classes have equal size.

  Args:
    x_sequence: A list of sequence features, and each element inside the list is
      a numpy.array of shape (sequence length, user embedding size).
    y_label: A numpy.array of sequence labels.
    class_size: An optional variable, which specifies the size for each class.
    shuffle: An optional variable, which decides whether to shuffle the
      generated dataset.

  Returns:
    Balanced dataset x_sequence, y_label in the same data structure as input.
  """
  unique_labels, count = np.unique(y_label, return_counts=True)
  class_size = int(min(np.min(count), class_size))
  indexes = [np.random.choice(np.where(y_label == l)[0], class_size)
             for l in unique_labels]
  indexes = np.hstack(indexes)
  if shuffle:
    np.random.shuffle(indexes)
  x_sequence = [x_sequence[index] for index in indexes]
  y_label = y_label[indexes]
  return x_sequence, y_label


def data_padding(x):
  """Pads and packs input sequence."""
  x_lengths = [len(seq) for seq in x]
  x_padded = rnn.pad_sequence(x, batch_first=True).float()
  x_packed = rnn.pack_padded_sequence(
      x_padded, x_lengths, batch_first=True, enforce_sorted=False)
  return x_packed


def load_bert_embeddings(model, tokenizer, title_list, pooling='average'):
  """Turns the post title lists into sequences of bert embedding.

  Uses pre-trained bert to generate embedding for each token in the post title,
  and applys average/max pooling to get title embedding.

  Args:
    model: Bert model
    tokenizer: Tokenizer for post title
    title_list: A list of post titles, and each title is a string
    pooling: An optional variable, which decides the pooling method

  Returns:
    A numpy.array of the sequence features, and its shape is
    (sequence length, bert embedding size).

  Raises:
    ValueError: If the value for the parameter is not correct.
  """
  if model is None:
    raise ValueError('The parameter model should not be None.')
  if tokenizer is None:
    raise ValueError('The parameter tokenizer should not be None.')
  emb_list = []
  with torch.no_grad():
    for title in title_list:
      input_ids = torch.tensor([tokenizer.encode(title)])
      if pooling == 'max':
        emb = model(input_ids)[0].max(axis=1)[0].numpy()[0]
      elif pooling == 'average':
        emb = model(input_ids)[0].mean(axis=1).numpy()[0]
      else:
        raise ValueError('The value of pooling is not supported.')
      emb_list.append(emb)
  return np.array(emb_list)


def load_bert_input_with_label(sequence_df, task, pooling='average'):
  """Loads sequence dataset with pre-trained bert embedding for baselines.

  Args:
    sequence_df: A pandas.dataframe containing the sequence dataset.
    task: A string of task name.
    pooling: An optional variable, which decides the pooling method

  Returns:
    x_sequence: A list of sequence features, and each element inside the list
      is a numpy.array of shape (sequence length, graph embedding size).
    y_label: A numpy.array of indexes in label_list for each sequence label.
    label_list: A list of unique labels.

  Raises:
    ValueError: If the value for the parameter is not correct.
  """
  if sequence_df is None:
    raise ValueError('The parameter sequence_df should not be None.')
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  model = BertModel.from_pretrained('bert-base-uncased')
  title_lists = sequence_df['title_list'].tolist()
  x_sequence = [load_bert_embeddings(model, tokenizer, title_list, pooling)
                for title_list in title_lists]
  if task == 'sr':
    y_label, label_list = pd.factorize(sequence_df['sr'])
  elif task in ['cat', 'fake']:
    y_label, label_list = pd.factorize(sequence_df['label'])
  else:
    raise ValueError('The value of task is not supported.')
  return x_sequence, y_label, label_list


def generate_user_list_and_edge_weight_dict(edgelist, post_author):
  """Generates user list and edge weight dictionary for the comment tree graph.

  Args:
    edgelist: A list of tuples and each tuple is an edge between two nodes. The
      nodes are user ids. The edgelist describes the comment tree graph within
      the post.
    post_author: A string denoting the author id of the post.

  Returns:
    user_list: A list of unique user ids within the comment tree graph. The
      author of the post is always at index 0.
    edge_weight_dict: A dictionary with keys as edge and value as weight.
      The nodes in the edge are the indexes of user ids in user_list.
  """
  user_list = [post_author]
  user_list.extend(list(set(
      [user for edge in edgelist  # pylint: disable=g-complex-comprehension
       for user in edge if user != post_author])))
  user_index_dict = {user: index for index, user in enumerate(user_list)}
  edge_weight_dict = collections.defaultdict(int)
  for edge in edgelist:
    edge_weight_dict[(user_index_dict[edge[0]],
                      user_index_dict[edge[1]])] += 1
  return user_list, edge_weight_dict


# TODO(huangdiana): add a test
def edge_weight_dict_to_adjacency_matrix(edge_weight_dict, user_list, id_list,
                                         weighted=False, weight_type='sum'):
  """Turns edge weight dictionary to processed adjacency matrix for GCN.

  GCN only supports undirected graphs. For weighted directed graph, generating
  the weight for undirected edge between two nodes v1, v2 has two ways:
  max: w_new(v1, v2) = max(w_old(v1, v2), w_old(v2, v1));
  sum: w_new(v1, v2) = w_old(v1, v2) + w_old(v2, v1). If there are nodes which
  don't have embeddings, they will be removed and the rest nodes will be
  re-indexed. If the post author doesn't have embedding, the post will be
  removed.

  Args:
    edge_weight_dict: A dictionary with keys as edge and value as weight.
      The nodes in the edge are the indexes of user ids in user_list.
    user_list: A list of users ids in the post comment tree.
    id_list: A list of user ids which have embedding in the post comment tree.
      If all users have embedding, id_list argument must be in the same
      order as user_list and id_list == user_list.
    weighted: An optional boolean variable, which decides whether the graph is
      weighted.
    weight_type: An optional variable, which decides the weighting type among
      max and sum when weighted equals True.

  Returns:
    adj: A spicy sparse matrix of normalized adjacency matrix with self loop.

  Raises:
    ValueError: If the value for the parameter is not correct.
  """
  num_users = len(id_list)
  if user_list[0] != id_list[0]:
    # TODO(huangdiana): assign random constant embedding to authors without
    # embeddings
    adj = None  # when post author doesn't have embedding
  else:
    if user_list == id_list:
      edge_weight_dict_update = edge_weight_dict
    else:
      user_list_index_dict = {user: index for index, user
                              in enumerate(user_list)}
      id_list_index_dict = {user: index for index, user in enumerate(id_list)}
      mapping = {user_list_index_dict[user]: id_list_index_dict[user]
                 for user in id_list}
      edge_weight_dict_update = {}
      for edge in edge_weight_dict:  # only add nodes with embeddings
        if edge[0] in id_list and edge[1] in id_list:
          edge_weight_dict_update[(mapping[edge[0]],
                                   mapping[edge[1]])] = edge_weight_dict[edge]
      for i in range(num_users):  # add self-loop
        edge_weight_dict_update[(i, i)] = 1

    edge_list = np.array([list(edge) for edge in edge_weight_dict_update])
    if weighted:
      weights = np.array(list(edge_weight_dict.values()))
    else:
      weights = np.ones(len(edge_list))

    adj = scipy.sparse.coo_matrix((weights, np.transpose(edge_list)),
                                  shape=(num_users, num_users))
    # build symmetric adjacency matrix
    if not weighted or (weighted and weight_type == 'max'):
      adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    elif weighted and weight_type == 'sum':
      adj = adj + adj.T
    else:
      raise ValueError('The value of weight_type is not supported.')
    adj = sklearn.preprocessing.normalize(adj, norm='l1', axis=1)
  return adj


def load_post_features_and_adjacency_matrix(post_list, df_post, embeddings_dict,
                                            weighted=False, weight_type='sum'):
  """Loads post sequence dataset with graph embedding and adjacency matrix.

  Args:
    post_list: A list of post author ids.
    df_post: A pandas.DataFrame for post data.
    embeddings_dict: A dictionary, where keys are user ids and values are
      their graph embedding.
    weighted: An optional variable, which decides whether the graph is weighted.
    weight_type: An optional variable, which decides the weighting type among
      max and sum when weighted equals True.

  Returns:
    post_data_sequence: A list of post data, and each element is a list
      [user feature matrix, adjacency matrix].
  """
  post_data_sequence = []
  for post_id in post_list:
    user_list, edge_weight_dict = df_post[
        df_post['post_id'] == post_id]['user_list_and_edgelist'].item()
    x_feature, id_list = load_embeddings(user_list, embeddings_dict)
    if x_feature is not None:
      adjacency_matrix = edge_weight_dict_to_adjacency_matrix(
          edge_weight_dict, user_list, id_list, weighted=weighted,
          weight_type=weight_type)
      if adjacency_matrix is not None:
        post_data_sequence.append([x_feature, adjacency_matrix])
  if not post_data_sequence:
    post_data_sequence = None
  return post_data_sequence


def load_gcn_rnn_input_with_label(sequence_df, df_post, embeddings_dict,
                                  weighted=False, weight_type='sum'):
  """Loads sequence dataset with graph embedding for GCN+RNN model.

  Args:
    sequence_df: A pandas.dataframe containing the sequence dataset.
    df_post: A pandas.DataFrame for post data.
    embeddings_dict: A dictionary, where keys are user ids and values are
      their graph embedding.
    weighted: An optional variable, which decides whether the graph is weighted.
    weight_type: An optional variable, which decides the weighting type among
      max and sum when weighted equals True.

  Returns:
    post_data_sequence: A list of post data, and each element is a list
      [user feature matrix, adjacency matrix].
  """
  if sequence_df is None:
    raise ValueError('The parameter sequence_df should not be None.')
  if df_post is None:
    raise ValueError('The parameter df_post should not be None.')
  if embeddings_dict is None:
    raise ValueError('The parameter embeddings_dict should not be None.')

  sequence_df['post_features_and_edgelist'] = sequence_df['post_list'].apply(
      lambda x:  # pylint: disable=g-long-lambda
      load_post_features_and_adjacency_matrix(
          x, df_post, embeddings_dict, weighted=weighted,
          weight_type=weight_type))
  sequence_df = sequence_df.dropna()
  x_sequence = sequence_df['post_features_and_edgelist'].tolist()
  y_label, label_list = pd.factorize(sequence_df['label'])
  return x_sequence, y_label, label_list


def sparse_matrix_to_sparse_tensor(matrix):
  """Turns a scipy sparse matrix into a torch sparse tensor.

  Args:
    matrix: A scipy.sparse matrix type among csr_matrix, csc_matrix
      and coo_matrix.

  Returns:
    A torch.sparse tensor of the input matrix.
  """
  matrix = matrix.tocoo().astype(np.float32)
  indices = torch.from_numpy(
      np.vstack((matrix.row, matrix.col)).astype(np.int64))
  values = torch.from_numpy(matrix.data)
  shape = torch.Size(matrix.shape)
  return torch.sparse.FloatTensor(indices, values, shape)


def data_into_tensor(x, y, batch, device):
  """Turns data into tensor for current batch for gcn+rnn model.

  Args:
    x: A list of url sequences containing input features. Each url sequence is a
      list of posts, and each post element is [x_feature,adjacency matrix].
    y: A list of labels.
    batch: a numpy array of indexed inside the current batch.
    device: A string denoting cpu or gpu.

  Returns:
    batch_x_return: A list of url sequences containing input features. Each url
      sequence is a list of posts, and each post element is a list of two
      tensors [x_feature, adjacency matrix]. x_feature has size
      (number of users in a comment graph, user embedding size) and the
      adjacency matrix is for the comment graph.
    batch_y_return: A tensor of labels.
  """
  batch_x = [x[idx] for idx in batch]
  batch_x_return = []
  for x_sequence in batch_x:
    each_url_input = []
    for x_feature, adjacency_matrix in x_sequence:
      each_url_input.append(
          [torch.from_numpy(x_feature).to(device),
           sparse_matrix_to_sparse_tensor(adjacency_matrix).to(device)])
    batch_x_return.append(each_url_input)
  batch_y_return = torch.from_numpy(y[batch]).to(device)
  return batch_x_return, batch_y_return


def save_model(model, optimizer, path, filename):
  """Saves model.

  Args:
    model: An instance of torch.nn.Module.
    optimizer: An instance of torch.optim.Optimizer.
    path: (String) Local path to save the model.
    filename: (String) The prefix of the file name for the saved model.
  """
  os.makedirs(path, exist_ok=True)
  torch.save(model.state_dict(), os.path.join(path, '%s_model.pt' % filename))
  torch.save({'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict()},
             os.path.join(path, '%s_checkpoint.tar' % filename))
