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

"""Generates sequence list from raw data."""


import os
import pickle

from absl import app
from absl import flags
import pandas as pd

import utils  # pylint: disable=g-bad-import-order

FLAGS = flags.FLAGS
# TODO(huangdiana, palowitch): make these proto enums instead.
flags.DEFINE_string(
    'task', 'cat', ('task: sr (subreddit classification), '
                    'cat (url categorization), fake (fake news detection)'))
flags.DEFINE_string(
    'return_type', 'user_sequence',
    ('user_sequence (a list of user ids), '
     'post_comment_graph_sequence (a list of post ids, plus post dataframe), '
     'post_comment_flat_sequence (a list of post authors and commenters), '
     'post_title_sequence (a list of post titles).'))
flags.DEFINE_integer('top', 30, 'number of top subreddits')
flags.DEFINE_string('name', '', 'specified name for the saved file.')
flags.DEFINE_string('data_file', '', 'raw data path')
flags.DEFINE_boolean('balancing', False, 'create balance_df for task fake')
flags.DEFINE_string('save_path', '', 'sequence save path')

pickle.HIGHEST_PROTOCOL = 4

# Flag specifications
flags.mark_flag_as_required('data_file')
flags.mark_flag_as_required('save_path')


def main(_):
  os.makedirs(FLAGS.save_path, exist_ok=True)
  df = pd.read_csv(FLAGS.data_file)

  if FLAGS.balancing and FLAGS.task == 'fake':
    df1 = df[['label', 'url']].drop_duplicates()
    size = int(df1.groupby('label').count().min())
    df_positive = df1[df1['label'] == 1].sample(size)
    df_negative = df1[df1['label'] == 0].sample(size)
    balance_df = pd.concat([df_positive, df_negative])
    balance_df.to_hdf(os.path.join(FLAGS.save_path, '%s_%s_%s.h5'%(
        FLAGS.task, 'url_balance', FLAGS.name)), key='df')

  if FLAGS.task == 'sr':
    if FLAGS.return_type != 'user_sequence':
      raise ValueError('return_type only supports user_sequence for task sr.')
    top_sr = df.groupby('sr').agg({'post_id': 'nunique'}).reset_index()
    top_sr = top_sr.nlargest(FLAGS.top, 'post_id')['sr']
    df = df.loc[df['sr'].isin(top_sr)]
    df_seq = df.groupby('post_id').apply(
        lambda x: list(x.sort_values('act_time')['user'])).reset_index(
            name='user_list')
    df_sr = df[['post_id', 'sr']].drop_duplicates()
    df_seq = df_seq.merge(df_sr, on='post_id', how='left')
    df_seq.to_hdf(os.path.join(
        FLAGS.save_path, '%s_%s_top%s_subreddit_%s.h5' % (
            FLAGS.task, FLAGS.return_type, FLAGS.top, FLAGS.name)), key='df')

  elif FLAGS.task in ['cat', 'fake']:
    if FLAGS.return_type == 'user_sequence':
      df1 = df[['post_created_time', 'post_author', 'url']].drop_duplicates()
      df_seq = df1.groupby('url').apply(
          lambda x: list(x.sort_values('post_created_time')['post_author'])
          ).reset_index(name='user_list')

    elif FLAGS.return_type == 'post_title_sequence':
      df1 = df[['post_created_time', 'post_title', 'url']].drop_duplicates()
      df_seq = df1.groupby('url').apply(
          lambda x: list(x.sort_values('post_created_time')['post_title'])
          ).reset_index(name='title_list')

    elif FLAGS.return_type == 'post_comment_graph_sequence':
      df1 = df[['post_created_time', 'post_id', 'url']].drop_duplicates()
      df_seq = df1.groupby('url').apply(
          lambda x: list(x.sort_values('post_created_time')['post_id'])
          ).reset_index(name='post_list')
      df_post = df.groupby('post_id').apply(
          lambda x: list(zip(x.commenter1, x.commenter2))
          ).reset_index(name='edgelist')
      df_post_author = df[['post_id', 'post_author']].drop_duplicates()
      df_post = df_post.merge(df_post_author, on='post_id', how='left')
      df_post['user_list_and_edgelist'] = df_post[
          ['edgelist', 'post_author']].apply(
              lambda x: utils.generate_user_list_and_edge_weight_dict(*x),
              axis=1)
      df_post = df_post[['post_id', 'user_list_and_edgelist']]
      df_post.to_hdf(os.path.join(FLAGS.save_path, 'post_%s_%s_%s.h5'%(
          FLAGS.task, FLAGS.return_type, FLAGS.name)), key='df')

    elif FLAGS.return_type == 'post_comment_flat_sequence':
      df_url_commenter = df[['url', 'commenter1',
                             'comment_created_time']].drop_duplicates()
      df_url_commenter.rename(
          {'commenter1': 'user', 'comment_created_time': 'created_time'},
          axis=1, inplace=True)
      df_url_post_author = df[['url', 'post_author',
                               'post_created_time']].drop_duplicates()
      df_url_post_author.rename(
          {'post_author': 'user', 'post_created_time': 'created_time'},
          axis=1, inplace=True)
      df_url_user = pd.concat([df_url_commenter, df_url_post_author])
      df_seq = df_url_user.groupby('url').apply(
          lambda x: list(x.sort_values('created_time')['user'])).reset_index(
              name='user_list')

    else:
      raise ValueError('return_type value is not supported for this task.')

    df_url_label = df[['url', 'label']].drop_duplicates()
    df_seq = df_seq.merge(df_url_label, on='url', how='left')
    df_seq.to_hdf(os.path.join(FLAGS.save_path, '%s_%s_%s.h5'%(
        FLAGS.task, FLAGS.return_type, FLAGS.name)), key='df')

  else:
    raise ValueError('task value is not supported.')

if __name__ == '__main__':
  app.run(main)
