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

"""Baselines for classification task."""


import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics
from sklearn import svm
import xgboost as xgb

if not __package__:
  import utils  # pylint: disable=g-bad-import-order,g-import-not-at-top
  import utils_gcs  # pylint: disable=g-bad-import-order,g-import-not-at-top
else:
  from gnns_for_news import utils  # pylint: disable=g-bad-import-order,g-import-not-at-top
  from gnns_for_news import utils_gcs  # pylint: disable=g-bad-import-order,g-import-not-at-top

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'task', 'cat', ('task: sr(subreddit classification), '
                    'cat(url categorization), fake(fake news detection)'))
flags.DEFINE_string('embedding', 'bert',
                    'embedding: bert(for post title), graph(for user)')
flags.DEFINE_string('gcs_path_in', None, 'gcs bucket input directory')
flags.DEFINE_string('gcs_path_out', None, 'gcs bucket output directory')
flags.DEFINE_string('local_path', './fake_input/', 'graph csv/gpickle file')
flags.DEFINE_string('g_emb', '', 'graph embedding file')
flags.DEFINE_string('seq_file', '', 'post sequence file')
flags.DEFINE_string('balance_df', '', 'the balanced dataset with url ids')
# Classification parameters
flags.DEFINE_string('model', 'xgboost', 'xgboost, svm')
flags.DEFINE_float('train_ratio', 0.7, 'training data ratio')
flags.DEFINE_integer('epochs', 10, 'number of epochs')

# Flag specifications
flags.mark_flag_as_required('gcs_path_in')
flags.mark_flag_as_required('gcs_path_out')
flags.mark_flag_as_required('g_emb')
flags.mark_flag_as_required('seq_file')


def logging_info_test(test_result):
  if len(test_result) == 2:
    logging.info(('Test Acc : %.4f | Test F1 : %.4f'), *test_result)
  else:
    logging.info(('Test Acc %.4f | Test Micro-F1 : %.4f | '
                  'Test Macro-F1 : %.4f | Test Weighted Macro-F1 : %.4f'),
                 *test_result)


def evaluate_test(y_true, y_pred):
  """Evaluates test data."""
  if len(set(y_true)) == 2:
    test_result = [metrics.accuracy_score(y_true, y_pred),
                   metrics.f1_score(y_true, y_pred)]
  else:
    test_result = [metrics.accuracy_score(y_true, y_pred),
                   metrics.f1_score(y_true, y_pred, average='micro'),
                   metrics.f1_score(y_true, y_pred, average='macro'),
                   metrics.f1_score(y_true, y_pred, average='weighted')]
  logging_info_test(test_result)
  return test_result


def main(_):
  if not os.path.exists(FLAGS.local_path):
    utils_gcs.download_files_from_gcs(FLAGS.local_path, FLAGS.gcs_path_in)
  logging.info('Data downloaded successfully!')

  sequence_df = pd.read_hdf(
      os.path.join(FLAGS.local_path, FLAGS.seq_file), 'df')
  if FLAGS.balance_df:
    balance_df = pd.read_hdf(
        os.path.join(FLAGS.local_path, FLAGS.balance_df), 'df')
    sequence_df = sequence_df[sequence_df['url'].isin(balance_df['url'])]
  if FLAGS.embedding == 'graph':
    embeddings_dict = utils.get_n2v_graph_embedding(
        os.path.join(FLAGS.local_path, FLAGS.g_emb), graph_gen=False,
        normalize_type='minmax')
    x_sequence, y_label, _ = utils.load_input_with_label(
        sequence_df, embeddings_dict, FLAGS.task)
  elif FLAGS.embedding == 'bert':
    x_sequence, y_label, _ = utils.load_bert_input_with_label(
        sequence_df, FLAGS.task, pooling='average')

  x_averaged = np.array([np.mean(seq, axis=0) for seq in x_sequence])

  # model training/testing
  logging.info('Classifier : %s', FLAGS.model)
  if FLAGS.model == 'svm':
    model = svm.SVC()
  elif FLAGS.model == 'xgboost':
    model = xgb.XGBClassifier()
  test_results = []
  for epoch in range(FLAGS.epochs):
    logging.info('Epoch %s', epoch)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x_averaged, y_label, train_size=FLAGS.train_ratio)
    model.fit(x_train, y_train)
    pred_y_test = model.predict(x_test)
    test_results.append(evaluate_test(y_test, pred_y_test))
  test_results = np.mean(np.array(test_results), axis=0)
  logging.info('Average results of %d epochs ', FLAGS.epochs)
  logging_info_test(test_results)

if __name__ == '__main__':
  app.run(main)
