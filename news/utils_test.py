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

"""Unit tests for data preprocessing utils."""

import unittest

from absl import app
import numpy as np

import utils  # pylint: disable=g-bad-import-order


class TestDataProcessingMethods(unittest.TestCase):

  def test_equal_binary(self):
    """Tests if binary data have balanced class size."""
    x_seq = [np.array([1, 2, 3]), np.array([1]), np.array([2]),
             np.array([1, 3]), np.array([3]), np.array([1, 3]),
             np.array([1, 2, 3]), np.array([2])]
    y_label = np.array([1, 0, 1, 1, 0, 1, 0, 1])
    x_seq, y_label = utils.balance_data(x_seq, y_label)
    self.assertEqual(sum(y_label)*2, len(y_label))

  def test_equal_multi(self):
    """Tests if multi-class data have balanced class size."""
    x_seq = [np.array([1, 2, 3]), np.array([1]), np.array([2]),
             np.array([1, 3]), np.array([3]), np.array([1, 3]),
             np.array([1, 2, 3]), np.array([2])]
    y_label = np.array([1, 0, 2, 1, 0, 2, 0, 1])
    x_seq, y_label = utils.balance_data(x_seq, y_label)
    self.assertEqual(np.sum(y_label == 0), 2)
    self.assertEqual(np.sum(y_label == 1), 2)
    self.assertEqual(np.sum(y_label == 2), 2)

  def test_equal_with_size(self):
    """Tests if binary data have balanced class size with specified size."""
    x_seq = [np.array([1, 2, 3]), np.array([1]), np.array([2]),
             np.array([1, 3]), np.array([3]), np.array([1, 3]),
             np.array([1, 2, 3]), np.array([2])]
    y_label = np.array([1, 0, 1, 1, 0, 1, 0, 1])
    x_seq, y_label = utils.balance_data(x_seq, y_label, class_size=2)
    self.assertEqual(np.sum(y_label == 0), 2)
    self.assertEqual(np.sum(y_label == 1), 2)

  def test_generate_user_list_and_edge_weight_dict(self):
    """Tests generate_user_list_and_edge_weight_dict."""
    edgelist = [('u1', 'u0'), ('u2', 'u0'), ('u3', 'u0'), ('u4', 'u2'),
                ('u5', 'u1'), ('u6', 'u1'), ('u2', 'u0'), ('u2', 'u4')]
    post_author = 'u0'
    user_list_expected = ['u0', 'u1', 'u2', 'u3', 'u4', 'u5', 'u6']
    user_list, edge_weight_dict = utils.generate_user_list_and_edge_weight_dict(
        edgelist, post_author)
    user_index_dict = {user: index for index, user in enumerate(user_list)}
    self.assertEqual(set(user_list), set(user_list_expected))
    self.assertEqual(user_list[0], post_author)
    self.assertEqual(
        edge_weight_dict[(user_index_dict['u1'], user_index_dict['u0'])], 1)
    self.assertEqual(
        edge_weight_dict[(user_index_dict['u2'], user_index_dict['u0'])], 2)
    self.assertEqual(
        edge_weight_dict[(user_index_dict['u3'], user_index_dict['u0'])], 1)
    self.assertEqual(
        edge_weight_dict[(user_index_dict['u4'], user_index_dict['u2'])], 1)
    self.assertEqual(
        edge_weight_dict[(user_index_dict['u5'], user_index_dict['u1'])], 1)
    self.assertEqual(
        edge_weight_dict[(user_index_dict['u6'], user_index_dict['u1'])], 1)
    self.assertEqual(
        edge_weight_dict[(user_index_dict['u2'], user_index_dict['u4'])], 1)

if __name__ == '__main__':
  app.run(unittest.main())
