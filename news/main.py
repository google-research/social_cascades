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

"""TraceMiner Node2Vec and RNN on Pytorch."""


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
# TODO(huangdiana): convert all default args for strings to ''.
flags.DEFINE_string('gcs_path_in', None, 'gcs bucket input directory')
flags.DEFINE_string('gcs_path_out', None, 'gcs bucket output directory')
flags.DEFINE_string('local_path_in', './fake_input/', 'local path for input')
flags.DEFINE_string('local_path_out', './fake_output/', 'local path for output')
flags.DEFINE_string('g_emb', None, 'graph embedding file')
flags.DEFINE_string('seq_file', None, 'post sequence file')
flags.DEFINE_string('balance_df', '', 'the balanced dataset with url ids')
# RNN, LSTM parameters
flags.DEFINE_string('model', 'rnn', 'rnn, lstm')
flags.DEFINE_float('train_ratio', 0.8, 'training data ratio')
flags.DEFINE_float('val_ratio', 0.1, 'validation data ratio')
flags.DEFINE_integer('batch_size', 64, 'bacth size for rnn')
flags.DEFINE_integer('dim', 128, 'graph embedding dimension')
flags.DEFINE_integer('hid_dim', 32, 'hidden dimension in RNN, LSTM')
flags.DEFINE_integer('num_layers', 2, 'number of layers in RNN, LSTM')
flags.DEFINE_boolean('bi', False, 'birectional')
flags.DEFINE_float('dropout', 0.8, ('dropout rate, will be auto-set to 0.0 if'
                                    'num_layers=1'))
flags.DEFINE_integer('epochs', 40, 'epochs')
flags.DEFINE_float('lr', 0.002, 'learning rate')
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


def print_gpu_info():
  """print gpu info."""
  logging.info('torch.__version__: %s', torch.__version__)
  logging.info('torch.cuda.device_count(): %s', torch.cuda.device_count())
  logging.info('torch.cuda.is_available(0): %s', torch.cuda.is_available())
  if torch.cuda.is_available():
    logging.info('torch.cuda.current_device(): %s', torch.cuda.current_device())
    logging.info('torch.cuda.get_device_name(0): %s',
                 torch.cuda.get_device_name(0))


def train(model_tm, x, y, train_batches, criterion, optimizer,
          device, print_step):
  """Trains Traceminer model.

  Args:
    model_tm: An instance of RNN or LSTM class.
    x: A list of url sequences containing post author embeddings. Each url
      sequence is a numpy.array of size (number of users, graph embedding size).
    y: A one-dimensional numpy.array of labels.
    train_batches: A list of one-dimension numpy.arrays. Each numpy.array has
      indexes of input x, label y for each batch during training.
    criterion: Model loss function.
    optimizer: Model optimizer.
    device: A string of device type, cpu or gpu.
    print_step: Number of steps for logging information.
  """
  model_tm.train()
  for i, batch in enumerate(train_batches):
    batch_x = [torch.from_numpy(x[idx]) for idx in batch]
    batch_x = utils.data_padding(batch_x).to(device)
    batch_y = torch.from_numpy(y[batch]).to(device)
    optimizer.zero_grad()
    output = model_tm(batch_x)
    loss = criterion(output, batch_y)
    loss.backward()
    optimizer.step()
    pred = nn.functional.log_softmax(output, 1)
    acc = metrics.accuracy_score(batch_y.data.cpu().numpy(),
                                 torch.argmax(pred, 1).data.cpu().numpy())
    if i % print_step == 0:
      logging.info('Train Batch [%s/%s] | Loss: %.4f | Train Acc : %.4f',
                   i, len(train_batches), loss.item(), acc)


def val(model_tm, x, y, val_batches, device):
  """Evaluates Traceminer model on validation dataset.

  Args:
    model_tm: An instance of RNN/LSTM class.
    x: A list of url sequences containing input features. Each url
      sequence is a numpy.array of size (number of users, user embedding).
    y: A one-dimensional numpy.array of labels.
    val_batches: A list of one-dimension numpy.arrays. Each numpy.array has
      indexes of input x, label y for each batch during validation.
    device: A string of device type, cpu or gpu.

  Returns:
    val_f1: F1 score for the validation dataset, used as metrics for
    hyperparamter searching.
  """
  model_tm.eval()
  pred_y, true_y = [], []
  with torch.no_grad():
    for batch in val_batches:
      batch_x = [torch.from_numpy(x[idx]) for idx in batch]
      batch_x = utils.data_padding(batch_x).to(device)
      batch_y = torch.from_numpy(y[batch]).to(device)
      output = model_tm(batch_x)
      pred_y += torch.argmax(output, 1).data.cpu().tolist()
      true_y += batch_y.data.cpu().tolist()
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


def test(model_tm, x, y, test_batches, device):
  """Evaluates Traceminer model on test dataset.

  Args:
    model_tm: An instance of RNN/LSTM class.
    x: A list of url sequences containing input features. Each url
      sequence is a numpy.array of size (number of users, user embedding).
    y: A one-dimensional numpy.array of labels.
    test_batches: A list of one-dimension numpy.arrays. Each numpy.array has
      indexes of input x, label y for each batch during testing.
    device: A string of device type, cpu or gpu.
  """
  model_tm.eval()
  pred_y, true_y = [], []
  with torch.no_grad():
    for batch in test_batches:
      batch_x = [torch.from_numpy(x[idx]) for idx in batch]
      batch_x = utils.data_padding(batch_x).to(device)
      batch_y = torch.from_numpy(y[batch]).to(device)
      output = model_tm(batch_x)
      pred_y += torch.argmax(output, 1).data.cpu().tolist()
      true_y += batch_y.data.cpu().tolist()
  if len(set(y)) == 2:
    logging.info(('Test Acc : %.4f | F1 : %.4f'),
                 metrics.accuracy_score(true_y, pred_y),
                 metrics.f1_score(true_y, pred_y))
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
  embeddings_dict = utils.get_n2v_graph_embedding(
      os.path.join(FLAGS.local_path_in, FLAGS.g_emb), graph_gen=False,
      normalize_type='minmax')
  x_sequence, y_label, label_list = utils.load_input_with_label(
      sequence_df, embeddings_dict, FLAGS.task)

  train_idx, val_idx, test_idx = utils.split_data_idx(
      len(x_sequence), FLAGS.train_ratio, FLAGS.val_ratio)
  train_batches = np.array_split(train_idx, len(train_idx) / FLAGS.batch_size)
  val_batches = np.array_split(val_idx, len(val_idx) / FLAGS.batch_size)
  test_batches = np.array_split(test_idx, len(test_idx) / FLAGS.batch_size)

  # model training/testing
  logging.info('FLAGS.epochs: %s', FLAGS.epochs)
  logging.info('FLAGS.batch_size: %s', FLAGS.batch_size)
  logging.info('FLAGS.learning_rate: %s', FLAGS.lr)

  dropout = 0.0 if FLAGS.num_layers == 1 else FLAGS.dropout

  print_gpu_info()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  logging.info('Current device is %s', device.type)
  if FLAGS.model == 'rnn':
    tm_model = models.RNN(
        in_dim=FLAGS.dim, hid_dim=FLAGS.hid_dim, num_label=len(label_list),
        num_layers=FLAGS.num_layers, dropout=dropout).to(device)
  elif FLAGS.model == 'lstm':
    tm_model = models.LSTM(
        in_dim=FLAGS.dim, hid_dim=FLAGS.hid_dim, num_label=len(label_list),
        num_layers=FLAGS.num_layers, dropout=dropout,
        bi_direct=FLAGS.bi).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(tm_model.parameters(),
                               lr=FLAGS.lr, weight_decay=1e-6)
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
    train(tm_model, x_sequence, y_label, train_batches, criterion,
          optimizer, device, FLAGS.print_step)
    val_f1 = val(tm_model, x_sequence, y_label, val_batches, device)
    test(tm_model, x_sequence, y_label, test_batches, device)
    if FLAGS.use_optimizer:
      elapsed_secs = int(time.time() - start_time)
      metric_list = [{'metric': 'valf1', 'value': float(val_f1)}]
      ml_client.report_intermediate_objective_value(epoch, elapsed_secs,
                                                    metric_list, trial_id)

  logging.info('Experiment finished.')

  if FLAGS.save_model:
    filename = '%s_%s_%s' % (FLAGS.task, FLAGS.model, FLAGS.name)
    utils.save_model(tm_model, optimizer, FLAGS.local_path_out, filename)
    utils_gcs.upload_files_to_gcs(local_folder=FLAGS.local_path_out,
                                  gcs_path=FLAGS.gcs_path_out)

if __name__ == '__main__':
  app.run(main)
