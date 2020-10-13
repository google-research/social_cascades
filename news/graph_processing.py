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

"""Graph processing script."""


import os

from absl import app
from absl import flags
from absl import logging
import networkx as nx
import pandas as pd

from utils import graph_filter_with_degree
from utils import load_graph_from_edgelist_csv

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'g_file',
    '../proj_Data/cat_data/test3/sr_timespan_post_graph-00000-of-00001.csv',
    'raw graph edgelist csv file')
flags.DEFINE_integer('low', 40, 'low degree threshold')
flags.DEFINE_integer('high', 80, 'high degree threshold')
flags.DEFINE_string('data_file', '', 'raw data path')
flags.DEFINE_string('filename', '', 'graph filename')
flags.DEFINE_string('save_path', '', 'graph save path')


def main(_):
  df = pd.read_csv(FLAGS.data_file)
  author_set = set(df['author'].unique())

  graph = load_graph_from_edgelist_csv(FLAGS.g_file)
  logging.info('Original Graph size: %d nodes, %d edges',
               graph.number_of_nodes(), graph.number_of_edges())
  graph = graph_filter_with_degree(graph, FLAGS.low, FLAGS.high, author_set)
  logging.info('Filtered Graph size: %d nodes, %d edges',
               graph.number_of_nodes(), graph.number_of_edges())
  nx.write_gpickle(graph, os.path.join(
      FLAGS.save_path, FLAGS.filename + '%s_%s.gpickle' %
      (FLAGS.low, FLAGS.high)))
  logging.info('Saved graph.')

if __name__ == '__main__':
  app.run(main)
