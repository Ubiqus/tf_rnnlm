# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
  Originally from Tensorflow v0.11
  Utilities for parsing text files.
  
  ---
  December, 2016 - pltrdy
  https://github.com/pltrdy/tf_rnnlm

  Note:
    We "manually" map <eos> with id=1 and reserve id=0 for padding

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import numpy as np
import tensorflow as tf

# Don't change those values. I would have serious side effects on the model. 
EOS = "<eos>"
IEOS=2

BOS = "<bos>"
IBOS=1

PAD = "<pad>"
IPAD=0

def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return f.read().replace("\n"," %s " % EOS).split()


def _build_vocab(filename):
  data = _read_words(filename)
  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  words = list(words)
  words.remove(EOS)
  word_to_id = dict(zip(words, range(3,len(words)+3)))
  word_to_id[BOS] = IBOS
  word_to_id[EOS] = IEOS
  word_to_id[PAD] = IPAD
  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


def raw_data(data_path=None, training=True, word_to_id=None):
  """Load raw data from data directory "data_path".

  Reads text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  The PTB dataset comes from Tomas Mikolov's webpage:

  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to producer.
  """
  train_data, valid_data, test_data = [], [], []
  if training:
    train_path = os.path.join(data_path, "train.txt")
    valid_path = os.path.join(data_path, "valid.txt")
    test_path = os.path.join(data_path, "test.txt")

    if word_to_id is None:
      word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    
    if os.path.isfile(test_path):
      test_data = _file_to_word_ids(test_path, word_to_id)
  
  else:
    if not word_to_id:
      raise ValueError("Must set 'word_to_id' when action is not 'train'")
    test_path = os.path.join(data_path, "test.txt")
    test_data = _file_to_word_ids(test_path, word_to_id)
    
  return [train_data, valid_data, test_data], word_to_id
