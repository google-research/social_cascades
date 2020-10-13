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

"""Util functions for Google Cloud Storage(GCS)."""


import os
import re

from absl import logging
from google.cloud import storage

GCS_PATH_PREFIX = 'gs://'


def parsing_gcs_path(gcs_path):
  """Parses gcs_bucket name into gcs_folder from gcs_path.

  Args:
    gcs_path: GCS path (example: 'gs://bucket/directory').

  Returns:
    gcs_bucket: A string giving GCS bucket (example: 'bucket').
    gcs_folder: A string giving GCS folder (example: 'directory').

  Raises:
    ValueError: If the value for the parameter is not correct.
  """
  if gcs_path is None:
    raise ValueError('The parameter gcs_path should not be None.')
  if GCS_PATH_PREFIX not in gcs_path:
    raise ValueError('gcs_path not in gs://bucket/directory format.')
  no_prefix = gcs_path.lstrip(GCS_PATH_PREFIX)
  splits = no_prefix.split('/')
  gcs_bucket, gcs_folder = splits[0], '/'.join(splits[1:])
  return gcs_bucket, gcs_folder


def upload_files_to_gcs(local_folder, gcs_path):
  """Uploads files to GCS.

  Args:
    local_folder: Local folder (example: 'temp/output').
    gcs_path: GCS path (example: 'gs://bucket/directory').

  Raises:
    ValueError: If the value for the parameter is not correct.
  """
  if local_folder is None:
    raise ValueError('The parameter local_folder should not be None')
  gcs_bucket, gcs_folder = parsing_gcs_path(gcs_path)
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(gcs_bucket)
  for directory_path, _, filenames in os.walk(local_folder):
    for name in filenames:
      filename = os.path.join(directory_path, name)
      blob = storage.Blob(os.path.join(gcs_folder, name), bucket)
      with open(filename, 'rb') as f:
        blob.upload_from_file(f)
      logging.info('[upload] blob path: %s', blob.path)
      logging.info('[upload] bucket path: gs://%s/%s', gcs_bucket, gcs_folder)


def download_files_from_gcs(local_folder, gcs_path, pattern=None):
  """Downloads files from GCS.

  Args:
    local_folder: Local folder (example: 'temp/output').
    gcs_path: GCS path (example: 'gs://bucket/directory').
    pattern: An optional variable denoting the file patterns (example: '*.h5').

  Raises:
    ValueError: If the value for the parameter is not correct.
  """
  os.makedirs(local_folder, exist_ok=True)
  gcs_bucket, gcs_folder = parsing_gcs_path(gcs_path)
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(gcs_bucket)
  blobs = bucket.list_blobs(prefix=gcs_folder)
  if pattern:
    regex = re.compile(pattern)
    blobs = [blob for blob in blobs if regex.match(os.path.basename(blob.name))]
  for blob in blobs:
    logging.info('[download] blob path: %s', blob.path)
    logging.info('[download] bucket path: gs://%s/%s', gcs_bucket, gcs_folder)
    local_path = os.path.join(local_folder, os.path.basename(blob.name))
    blob.download_to_filename(local_path)
