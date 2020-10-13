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

"""Generate Graph Embedding using Node2Vec."""


import os

from absl import app
from absl import flags
from absl import logging

import utils  # pylint: disable=g-bad-import-order
import utils_gcs  # pylint: disable=g-bad-import-order


FLAGS = flags.FLAGS
flags.DEFINE_string('gcs_path_in', None, 'gcs bucket input directory')
flags.DEFINE_string('gcs_path_out', None, 'gcs bucket output directory')
flags.DEFINE_string('local_path', './fake_input/', 'graph csv/gpickle file')
# Graph embedding parameters
flags.DEFINE_string('load_method', 'csv', 'csv, gpickle')
flags.DEFINE_string('g_file', '', 'graph csv/gpickle file')
flags.DEFINE_integer('dim', 128, 'graph embedding dimension')
flags.DEFINE_integer('walk_len', 30, 'walk length')
flags.DEFINE_integer('num_walk', 100, 'number of walks')
flags.DEFINE_integer('workers', 4, 'number of workers')
flags.DEFINE_integer('win_size', 10, 'window size')
flags.DEFINE_string('g_emb', '', 'graph embedding file')


def main(_):
  if not os.path.exists(FLAGS.local_path):
    utils_gcs.download_files_from_gcs(FLAGS.local_path, FLAGS.gcs_path_in)
  logging.info('Data downloaded successfully!')
  graph = utils.load_graph(os.path.join(FLAGS.local_path, FLAGS.g_file),
                           load_method=FLAGS.load_method, directed=False)
  _ = utils.get_n2v_graph_embedding(
      os.path.join(FLAGS.local_path, FLAGS.g_emb), graph_gen=True,
      graph=graph, dimensions=FLAGS.dim, walk_length=FLAGS.walk_len,
      num_walks=FLAGS.num_walk, workers=FLAGS.workers,
      win_size=FLAGS.win_size, normalize_type='minmax')
  utils_gcs.upload_files_to_gcs(local_folder=FLAGS.local_path,
                                gcs_path=FLAGS.gcs_path_out)

if __name__ == '__main__':
  app.run(main)

