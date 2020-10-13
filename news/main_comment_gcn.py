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

"""GCN+RNN post sequence model on Pytorch."""


import json
import os
import time

from absl import app
from absl import flags
from absl import logging
import numpy as np
import pandas as pd
from sklearn import metrics
from tensorflow_enterprise_addons.cloudtuner import optimizer_client
import torch
from torch import nn

if not __package__:
  import models  # pylint: disable=g-bad-import-order,g-import-not-at-top
  import utils  # pylint: disable=g-bad-import-order,g-import-not-at-top
  import utils_gcs  # pylint: disable=g-bad-import-order,g-import-not-at-top
else:
  from gnns_for_news import models  # pylint: disable=g-bad-import-order,g-import-not-at-top
  from gnns_for_news import utils  # pylint: disable=g-bad-import-order,g-import-not-at-top
  from gnns_for_news import utils_gcs  # pylint: disable=g-bad-import-order,g-import-not-at-top

FLAGS = flags.FLAGS
flags.DEFINE_string('task', 'cat', ('task: sr(subreddit classification), '
                                    'cat(url categorization), '
                                    'fake(fake news detection)'))
flags.DEFINE_string('gcs_path_in', None, 'gcs bucket input directory')
flags.DEFINE_string('gcs_path_out', None, 'gcs bucket output directory')
flags.DEFINE_string('local_path_in', './fake_input/', 'local path for input')
flags.DEFINE_string('local_path_out', './fake_output/', 'local path for output')
flags.DEFINE_string('g_emb', '', 'graph embedding file')
flags.DEFINE_string('seq_file', '', 'post sequence file')
flags.DEFINE_string('post_file', '', 'post dataframe h5 file')
flags.DEFINE_string('balance_df', '', 'the balanced dataset with url ids')
# GCNRNN parameters
flags.DEFINE_string('model', 'rnn', 'rnn, lstm')
flags.DEFINE_float('train_ratio', 0.8, 'training data ratio')
flags.DEFINE_float('val_ratio', 0.1, 'validation data ratio')
flags.DEFINE_integer('batch_size', 16, 'bacth size for rnn')
flags.DEFINE_integer('dim', 128, 'graph embedding dimension')
flags.DEFINE_integer('gcn_hid_dim', 32, 'hidden dimension in GCN')
flags.DEFINE_integer('gcn_out_dim', 32, 'output dimension in GCN')
flags.DEFINE_integer('rnn_hid_dim', 32, 'hidden dimension in RNN/LSTM')
flags.DEFINE_integer('rnn_num_layers', 2, 'number of layers in RNN/LSTM')
flags.DEFINE_boolean('bi', False, 'birectional')
flags.DEFINE_float('rnn_dropout', 0.3, ('dropout rate for RNN/LSTM, will be'
                                        'auto-set to 0.0 if rnn_num_layers=1'))
flags.DEFINE_float('gcn_dropout', 0.3, 'dropout rate for GCN')
flags.DEFINE_boolean('gcn_bias', True, 'whether to have bias in GCN')
flags.DEFINE_integer('epochs', 20, 'epochs')
flags.DEFINE_float('lr_gcn', 0.0001, 'gcn learning rate')
flags.DEFINE_float('lr_rnn', 0.001, 'rnn learning rate')
flags.DEFINE_integer('print_step', 10, 'print step during training')
flags.DEFINE_boolean('save_model', False, 'save model')
flags.DEFINE_string('name', '', 'specify mode name')
# XCloud parameters
flags.DEFINE_string('trial_name', None,
                    'Identifying the current job trial for addMeasurement')
flags.DEFINE_string('study_config', None,
                    'Study configuration for CAIP Optimizer service')
flags.DEFINE_boolean('use_optimizer', False,
                     'Use dynamic job generation for hparam optimization')

# Flag specifications
flags.mark_flag_as_required('gcs_path_in')
flags.mark_flag_as_required('gcs_path_out')
flags.mark_flag_as_required('g_emb')
flags.mark_flag_as_required('seq_file')
flags.mark_flag_as_required('post_file')


def print_gpu_info():
  """print gpu info."""
  logging.info('torch.__version__: %s', torch.__version__)
  logging.info('torch.cuda.device_count(): %s', torch.cuda.device_count())
  logging.info('torch.cuda.is_available(0): %s', torch.cuda.is_available())
  if torch.cuda.is_available():
    logging.info('torch.cuda.current_device(): %s', torch.cuda.current_device())
    logging.info('torch.cuda.get_device_name(0): %s',
                 torch.cuda.get_device_name(0))


def train(gcn_rnn_model, x, y, train_batches, criterion, optimizer_gcn,
          optimizer_rnn, device, print_step):
  """Trains GCN+RNN post sequence model.

  Args:
    gcn_rnn_model: An instance of GCNRNN class.
    x: A list of url sequences containing input features. Each url
      sequence is a list of posts, and each post element is [x_feature,
      adjacency matrix]. x_feature is a numpy.array of size
      (number of users in a comment graph, user embedding size) and the
      adjacency matrix is a scipy sparse matrix for the comment graph.
    y: A one-dimensional numpy.array of labels.
    train_batches: A list of one-dimension numpy.arrays. Each numpy.array has
      indexes of input x, label y for each batch during training.
    criterion: Model loss function.
    optimizer_gcn: gcn optimizer.
    optimizer_rnn: rnn optimizer.
    device: A string of device type, cpu or gpu.
    print_step: Number of steps for logging information.
  """
  gcn_rnn_model.train()
  for i, batch in enumerate(train_batches):
    batch_x, batch_y = utils.data_into_tensor(x, y, batch, device)
    output = gcn_rnn_model(batch_x)
    optimizer_gcn.zero_grad()
    optimizer_rnn.zero_grad()
    loss = criterion(output, batch_y)
    loss.backward()
    optimizer_gcn.step()
    optimizer_rnn.step()
    pred = nn.functional.log_softmax(output, 1)
    acc = metrics.accuracy_score(batch_y.data.cpu().numpy(),
                                 torch.argmax(pred, 1).data.cpu().numpy())
    if i % print_step == 0:
      logging.info('Train Batch [%s/%s] | Loss: %.4f | Train Acc : %.4f',
                   i, len(train_batches), loss.item(), acc)
    del batch_x, batch_y
    torch.cuda.empty_cache()


def val(gcn_rnn_model, x, y, val_batches, device):
  """Evaluates GCN+RNN model on validation dataset.

  Args:
    gcn_rnn_model: An instance of GCNRNN class.
    x: A list of url sequences containing input features. Each url
      sequence is a list of posts, and each post element is [x_feature,
      adjacency matrix]. x_feature is a numpy.array of size
      (number of users in a comment graph, user embedding size) and the
      adjacency matrix is a scipy sparse matrix for the comment graph.
    y: A one-dimensional numpy.array of labels.
    val_batches: A list of one-dimension numpy.arrays. Each numpy.array has
      indexes of input x, label y for each batch during validation.
    device: A string of device type, cpu or gpu.

  Returns:
    val_f1: F1 score for the validation dataset, used as metrics for
      hyperparamter searching.
  """
  gcn_rnn_model.eval()
  pred_y, true_y = [], []
  with torch.no_grad():
    for batch in val_batches:
      batch_x, batch_y = utils.data_into_tensor(x, y, batch, device)
      output = gcn_rnn_model(batch_x)
      pred = nn.functional.log_softmax(output, 1)
      pred_y += torch.argmax(pred, 1).data.cpu().tolist()
      true_y += batch_y.data.cpu().tolist()
      del batch_x, batch_y
      torch.cuda.empty_cache()
  if len(set(y)) == 2:
    logging.info(('Validation Acc : %.4f | F1 : %.4f'),
                 metrics.accuracy_score(true_y, pred_y),
                 metrics.f1_score(true_y, pred_y))
    val_f1 = metrics.f1_score(true_y, pred_y)
  else:
    logging.info(('Validation Acc : %.4f | Validation Micro-F1 : %.4f | '
                  'Macro-F1 : %.4f | Weighted Macro-F1 : %.4f'),
                 metrics.accuracy_score(true_y, pred_y),
                 metrics.f1_score(true_y, pred_y, average='micro'),
                 metrics.f1_score(true_y, pred_y, average='macro'),
                 metrics.f1_score(true_y, pred_y, average='weighted'))
    val_f1 = metrics.f1_score(true_y, pred_y, average='macro')
  return val_f1


def test(gcn_rnn_model, x, y, test_batches, device):
  """Testing evalution for the GCN+RNN post sequence model.

  Args:
    gcn_rnn_model: An instance of GCNRNN class.
    x: A list of url sequences containing input features. Each url
      sequence is a list of posts, and each post element is [x_feature,
      adjacency matrix]. x_feature is a numpy.array of size
      (number of users in a comment graph, user embedding size) and the
      adjacency matrix is a scipy sparse matrix for the comment graph.
    y: A one-dimensional numpy.array of labels.
    test_batches: A list of one-dimension numpy.arrays. Each numpy.array has
      indexes of input x, label y for each batch during testing.
    device: A string of device type, cpu or gpu.
  """
  gcn_rnn_model.eval()
  pred_y, true_y = [], []
  with torch.no_grad():
    for batch in test_batches:
      batch_x, batch_y = utils.data_into_tensor(x, y, batch, device)
      output = gcn_rnn_model(batch_x)
      pred = nn.functional.log_softmax(output, 1)
      pred_y += torch.argmax(pred, 1).data.cpu().tolist()
      true_y += batch_y.data.cpu().tolist()
      del batch_x, batch_y
      torch.cuda.empty_cache()
  if len(set(y)) == 2:
    logging.info(('Test Acc : %.4f | F1 : %.4f'),
                 metrics.accuracy_score(true_y, pred_y),
                 metrics.f1_score(true_y, pred_y))
    print(metrics.confusion_matrix(true_y, pred_y))
  else:
    logging.info(('Test Acc : %.4f | Test Micro-F1 : %.4f | '
                  'Test Macro-F1 : %.4f | Test Weighted Macro-F1 : %.4f'),
                 metrics.accuracy_score(true_y, pred_y),
                 metrics.f1_score(true_y, pred_y, average='micro'),
                 metrics.f1_score(true_y, pred_y, average='macro'),
                 metrics.f1_score(true_y, pred_y, average='weighted'))


def main(_):
  if not os.path.exists(FLAGS.local_path_in) or FLAGS.use_optimizer:
    utils_gcs.download_files_from_gcs(FLAGS.local_path_in, FLAGS.gcs_path_in)
  logging.info('Data downloaded successfully!')

  sequence_df = pd.read_hdf(
      os.path.join(FLAGS.local_path_in, FLAGS.seq_file), 'df')
  if FLAGS.balance_df:
    balance_df = pd.read_hdf(
        os.path.join(FLAGS.local_path_in, FLAGS.balance_df), 'df')
    sequence_df = sequence_df[sequence_df['url'].isin(balance_df['url'])]
  df_post = pd.read_hdf(
      os.path.join(FLAGS.local_path_in, FLAGS.post_file), 'df')
  embeddings_dict = utils.get_n2v_graph_embedding(
      os.path.join(FLAGS.local_path_in, FLAGS.g_emb), graph_gen=False,
      normalize_type='minmax')
  x_sequence, y_label, label_list = utils.load_gcn_rnn_input_with_label(
      sequence_df, df_post, embeddings_dict)

  train_idx, val_idx, test_idx = utils.split_data_idx(
      len(x_sequence), FLAGS.train_ratio, FLAGS.val_ratio)
  train_batches = np.array_split(train_idx, len(train_idx) / FLAGS.batch_size)
  val_batches = np.array_split(val_idx, len(val_idx) / FLAGS.batch_size)
  test_batches = np.array_split(test_idx, len(test_idx) / FLAGS.batch_size)

  # model training/testing
  logging.info('FLAGS.epochs: %s', FLAGS.epochs)
  logging.info('FLAGS.batch_size: %s', FLAGS.batch_size)
  logging.info('FLAGS.lr_gcn: %s', FLAGS.lr_gcn)
  logging.info('FLAGS.lr_rnn: %s', FLAGS.lr_rnn)

  print_gpu_info()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  logging.info('Current device is %s', device.type)

  rnn_dropout = 0.0 if FLAGS.rnn_num_layers == 1 else FLAGS.rnn_dropout

  gcn_rnn_model = models.GCNRNN(
      gcn_in_dim=FLAGS.dim, gcn_hid_dim=FLAGS.gcn_hid_dim,
      gcn_out_dim=FLAGS.gcn_out_dim, rnn_type=FLAGS.model.upper(),
      rnn_hid_dim=FLAGS.rnn_hid_dim, rnn_out_dim=len(label_list),
      rnn_num_layers=FLAGS.rnn_num_layers, rnn_dropout=rnn_dropout,
      bi_direct=FLAGS.bi, gcn_dropout=FLAGS.gcn_dropout,
      gcn_bias=FLAGS.gcn_bias).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer_gcn = torch.optim.Adam(gcn_rnn_model.gcn.parameters(),
                                   lr=FLAGS.lr_gcn, weight_decay=1e-6)
  optimizer_rnn = torch.optim.Adam(gcn_rnn_model.rnn.parameters(),
                                   lr=FLAGS.lr_rnn, weight_decay=1e-6)
  if FLAGS.use_optimizer:
    # example trial_name:
    # 'projects/{project_id}/locations/{region}/studies/{study_id}/trials/{trial_id}'
    trial_name_split = FLAGS.trial_name.split('/')
    project_id = trial_name_split[1]
    region = trial_name_split[3]
    study_id = trial_name_split[-3]
    trial_id = trial_name_split[-1]
    logging.info('project_id: %s, region: %s, study_id: %s, trial_id: %s',
                 project_id, region, study_id, trial_id)
    ml_client = optimizer_client.create_or_load_study(
        project_id, region, study_id, json.loads(FLAGS.study_config))
  for epoch in range(FLAGS.epochs):
    logging.info('Epoch %s', epoch)
    start_time = time.time()
    train(gcn_rnn_model, x_sequence, y_label, train_batches, criterion,
          optimizer_gcn, optimizer_rnn, device, FLAGS.print_step)
    val_f1 = val(gcn_rnn_model, x_sequence, y_label, val_batches, device)
    test(gcn_rnn_model, x_sequence, y_label, test_batches, device)
    if FLAGS.use_optimizer:
      elapsed_secs = int(time.time() - start_time)
      metric_list = [{'metric': 'valf1', 'value': float(val_f1)}]
      ml_client.report_intermediate_objective_value(epoch, elapsed_secs,
                                                    metric_list, trial_id)

  logging.info('Experiment finished.')

  if FLAGS.save_model:
    filename = '%s_gcnrnn_%s_%s' % (FLAGS.task, FLAGS.model, FLAGS.name)
    utils.save_model(
        gcn_rnn_model, optimizer_gcn, FLAGS.local_path_out, filename + 'gcn')
    utils.save_model(
        gcn_rnn_model, optimizer_rnn, FLAGS.local_path_out, filename + 'rnn')
    utils_gcs.upload_files_to_gcs(local_folder=FLAGS.local_path_out,
                                  gcs_path=FLAGS.gcs_path_out)

if __name__ == '__main__':
  app.run(main)
