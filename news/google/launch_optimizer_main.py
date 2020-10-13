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

"""Launcher for main.py on PyTorch with GPUs using JobTrialGenerator.

Note that JobTrialGenerator uses CAIP Optimizer for automatic hyperparameter
tuning, which requires the training executable to report measurements via
setting up a CAIP Optimizer client.
"""


import os

from absl import app
from absl import flags
import termcolor

from google3.learning.brain.frameworks import xcloud as xm
from google3.learning.deepmind.xmanager import hyper
from google3.learning.vizier.service import automated_stopping_pb2
from google3.learning.vizier.service import vizier_pb2

GCS_PATH_PREFIX = 'gs://'

FLAGS = flags.FLAGS
flags.DEFINE_string('project_name', 'traceminer', 'name for the project')
flags.DEFINE_string('image_uri', None,
                    'A URI to a prebuilt Docker image, including tag.')
flags.DEFINE_string('base_image', None,
                    'A URI to a prebuilt Docker image, for option2.')
flags.DEFINE_boolean('use_gpu', True, 'use GPU')
flags.DEFINE_string('acc_type', 'v100', 'Accelerator type, v100 or t4')
flags.DEFINE_integer('num_gpus', 1, 'Number of GPUs.')
flags.DEFINE_string('gcs_path_in', None,
                    ('A GCS directory within a bucket to store input'
                     'in gs://bucket/directory format.'))
flags.DEFINE_string('gcs_path_out', None,
                    ('A GCS directory within a bucket to store output '
                     'in gs://bucket/directory format.'))
flags.DEFINE_string('task', 'cat', ('task: sr(subreddit classification), '
                                    'cat(url categorization), '
                                    'fake(fake news detection)'))
flags.DEFINE_string('local_path_in', './fake_input/', 'local path for input')
flags.DEFINE_string('local_path_out', './fake_output/', 'local path for output')
flags.DEFINE_string('g_emb', '', 'graph embedding file')
flags.DEFINE_string('seq_file', '', 'post sequence file')
flags.DEFINE_string('balance_df', '', 'the balanced dataset with url ids')
# RNN, LSTM parameters
flags.DEFINE_string('model', 'rnn', 'rnn, lstm')
flags.DEFINE_float('train_ratio', 0.8, 'training data ratio')
flags.DEFINE_float('val_ratio', 0.1, 'validation data ratio')
flags.DEFINE_integer('batch_size', 64, 'bacth size for rnn')
flags.DEFINE_integer('hid_dim', 32, 'hidden dimension in RNN, LSTM')
flags.DEFINE_integer('num_layers', 2, 'number of layers in RNN, LSTM')
flags.DEFINE_boolean('bi', False, 'birectional')
flags.DEFINE_float('dropout', 0.8, 'dropout')
flags.DEFINE_integer('epochs', 40, 'epochs')
flags.DEFINE_float('lr', 0.002, 'learning rate')
flags.DEFINE_integer('print_step', 10, 'print step during training')
flags.DEFINE_boolean('save_model', False, 'save model')
flags.DEFINE_string('name', '', 'specify model name')

# Flag specifications
flags.mark_flag_as_required('gcs_path_in')
flags.mark_flag_as_required('gcs_path_out')
flags.register_validator('gcs_path_in', lambda value: GCS_PATH_PREFIX in value,
                         message=('--gcs_path_in must follow'
                                  'gs://bucket/directory format'))
flags.register_validator('gcs_path_out', lambda value: GCS_PATH_PREFIX in value,
                         message=('--gcs_path_out must follow'
                                  'gs://bucket/directory format'))


def main(_):
  if FLAGS.use_gpu:
    accelerator = xm.GPU('nvidia-tesla-' + FLAGS.acc_type.lower(),
                         FLAGS.num_gpus)
  else:
    accelerator = None
  runtime = xm.CloudRuntime(
      cpu=3,
      memory=24,
      accelerator=accelerator,
  )

  args = {
      'task': FLAGS.task,
      'gcs_path_in': FLAGS.gcs_path_in,
      'gcs_path_out': FLAGS.gcs_path_out,
      'local_path_in': FLAGS.local_path_in,
      'local_path_out': FLAGS.local_path_out,
      'g_emb': FLAGS.g_emb,
      'seq_file': FLAGS.seq_file,
      'balance_df': FLAGS.balance_df,
      'train_ratio': FLAGS.train_ratio,
      'val_ratio': FLAGS.val_ratio,
      'bi': FLAGS.bi,
      'dropout': FLAGS.dropout,
      'print_step': FLAGS.print_step,
      'save_model': FLAGS.save_model,
      'name': FLAGS.name,
      'use_optimizer': True
  }

  if FLAGS.image_uri:
    # Option 1 This will use a user-defined docker image.
    executable = xm.CloudDocker(
        name=FLAGS.project_name,
        runtime=runtime,
        image_uri=FLAGS.image_uri,
        args=args,
    )
  else:
    # Option 2 This will build a docker image for the user. Set up environment.
    executable = xm.CloudPython(
        name=FLAGS.project_name,
        runtime=runtime,
        project_path=(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
        module_name='gnns_for_news.main',
        base_image=FLAGS.base_image,
        args=args,
        build_steps=(xm.steps.default_build_steps('gnns_for_news')),
    )
  # Set UNIT_LOG_SCALE to explore more values in the lower range
  # Set UNIT_REVERSE_LOG_SCALE to explore more values in the higher range
  parameters = [
      hyper.get_vizier_parameter_config(
          'model', hyper.categorical(['rnn', 'lstm'])),
      hyper.get_vizier_parameter_config(
          'batch_size', hyper.discrete([16 * k for k in range(1, 6)])),
      hyper.get_vizier_parameter_config(
          'hid_dim', hyper.discrete([16 * k for k in range(3, 10)])),
      hyper.get_vizier_parameter_config(
          'num_layers', hyper.discrete([1, 2])),
      hyper.get_vizier_parameter_config(
          'lr', hyper.interval(0.00001, 0.2), scaling='UNIT_LOG_SCALE'),
      hyper.get_vizier_parameter_config(
          'dropout', hyper.discrete([0.0, 0.15, 0.3, 0.5, 0.7])),
      hyper.get_vizier_parameter_config(
          'epochs', hyper.discrete([5, 10, 20, 30]))
  ]
  vizier_study_config = vizier_pb2.StudyConfig()
  for parameter in parameters:
    vizier_study_config.parameter_configs.add().CopyFrom(parameter)
  metric = vizier_study_config.metric_information.add()
  metric.name = 'valf1'
  metric.goal = vizier_pb2.StudyConfig.GoalType.Value('MAXIMIZE')
  # None early stopping
  early_stopping = automated_stopping_pb2.AutomatedStoppingConfig()
  vizier_study_config.automated_stopping_config.CopyFrom(early_stopping)

  exploration = xm.HyperparameterOptimizer(
      executable=executable,
      max_num_trials=128,
      parallel_evaluations=8,
      vizier_study_config=vizier_study_config
  )
  xm.launch(xm.ExperimentDescription(FLAGS.project_name), exploration)

  no_prefix = FLAGS.gcs_path_out.lstrip(GCS_PATH_PREFIX)
  print()
  print('When your job completes, you will see artifacts in ' +
        termcolor.colored(
            f'https://pantheon.corp.google.com/storage/browser/{no_prefix}',
            color='blue'))

if __name__ == '__main__':
  app.run(main)
