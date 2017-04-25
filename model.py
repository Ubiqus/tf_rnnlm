#!/usr/bin/env python

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

from __future__ import division
from __future__ import print_function

import math
import time
import json
import numpy as np
import tensorflow as tf
import sys


class Model(object):
  """The model."""

  def __init__(self, config, is_training=True, loss_fct="softmax", test_opti=False, use_fp16=False):
    self.config = config
    self.loss_fct = loss_fct
    self.is_training = is_training
    self.use_fp16=use_fp16

    
    self._build_model()

  def _build_model(self):
    batch_size = self.batch_size
    hidden_size = self.hidden_size
    vocab_size = self.vocab_size
    num_layers = self.num_layers
    keep_prob = self.keep_prob
    is_training = self.is_training
    data_type = self.data_type

    _inputs = tf.placeholder(tf.int32, [batch_size, None], "inputs")
    _targets = tf.placeholder(tf.int32, [batch_size, None], "targets")
    
    
    lstm_creator = lambda: tf.contrib.rnn.BasicLSTMCell(
                                        hidden_size, 
                                        forget_bias=0.0, state_is_tuple=True,
                                        reuse=True)
    import inspect
    def lstm_cell():
      # With the latest TensorFlow source code (as of Mar 27, 2017),
      # the BasicLSTMCell will need a reuse parameter which is unfortunately not
      # defined in TensorFlow 1.0. To maintain backwards compatibility, we add
      # an argument check here:
      size = hidden_size
      if 'reuse' in inspect.getargspec(
          tf.contrib.rnn.BasicLSTMCell.__init__).args:
        return tf.contrib.rnn.BasicLSTMCell(
            size, forget_bias=0.0, state_is_tuple=True,
            reuse=tf.get_variable_scope().reuse)
      else:
        return tf.contrib.rnn.BasicLSTMCell(
            size, forget_bias=0.0, state_is_tuple=True)

    lstm_creator = lstm_cell
    if is_training and keep_prob < 1:
      cell_creator = lambda:tf.contrib.rnn.DropoutWrapper(
          lstm_creator(), output_keep_prob=keep_prob)
    else:
      cell_creator = lstm_creator

    cell = tf.contrib.rnn.MultiRNNCell([cell_creator() for _ in range(num_layers)], state_is_tuple=True)
    _initial_state = cell.zero_state(batch_size, data_type)
    
    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, hidden_size], dtype=data_type)
      inputs = tf.nn.embedding_lookup(embedding, _inputs)

    if is_training and keep_prob < 1:
      inputs = tf.nn.dropout(inputs, keep_prob)
   
    # if num_steps == 0 we are in 'sentence mode' aka. dynamic
    if self.config.num_steps == 0:
      _mask = tf.sign(tf.to_float(_inputs))
    else:
      _mask = tf.ones([self.batch_size, self.config.num_steps])
    _seq_len = tf.reduce_sum(_mask, reduction_indices=1)

    # outputs is [bs x ts x hidden_size] (ts may be None)
    _outputs, state = tf.nn.dynamic_rnn(cell=cell, inputs=inputs,
        initial_state=_initial_state,
        sequence_length=_seq_len)

    _mask = tf.reshape(_mask, [-1])
   
    # output: [bs*ts x hidden_size]
    _output = tf.reshape(_outputs, [-1, hidden_size])


    self.inputs = _inputs
    self.targets= _targets

    self.mask = _mask
    self.seq_len = _seq_len
    self.output = _output

    loss, logits = self.compute_loss()
    _cost = tf.reduce_sum(loss) / batch_size

    self._final_state = state 
    self._initial_state = _initial_state
    self.loss, self.logits = loss, logits
    self.cost = _cost
   
    if logits is not None:
      elems = tf.range(vocab_size)
      self.choices = tf.multinomial(logits, 1)
    
    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(_cost, tvars),
                                      self.config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def compute_loss(self):
    fct = self.loss_fct
    fast_test = self.fast_test
    vocab_size = self.vocab_size
    hidden_size = self.hidden_size
    
    # TF 1.1
    _sequence_loss_by_example = tf.contrib.legacy_seq2seq.sequence_loss_by_example

    loss, logits = None, None

    if self.is_training or not (fast_test and fct == "softmax"):
      self.w = tf.get_variable("w", [vocab_size, hidden_size], dtype=self.data_type)
      self.b = tf.get_variable("b", [vocab_size], dtype=self.data_type)
      
      if fct == "softmax":
        # Softmax uses transposed weights which is very slow.
        # See 'transpose.py' for more information about fast_test
        self.w_t = tf.transpose(self.w)
    else:
      # The fast test tricks uses a Model saved with w_t instead of w
      # See 'transpose.py' to transpose your models in order to use fast test
      self.w_t = tf.get_variable("w_t", [hidden_size, vocab_size], dtype=self.data_type)
      self.b = tf.get_variable("b", [vocab_size], dtype=self.data_type)
    
    if fct == "softmax":
      logits = tf.matmul(self.output, self.w_t)+self.b
      loss = tf.nn.seq2seq.sequence_loss_by_example(
          [logits],
          [tf.reshape(self.targets, [-1])],
          [self.mask])
    
    elif fct == "sampledsoftmax":
      def _loss_fct(inputs_, labels_):
        labels_ = tf.reshape(labels_, [-1, 1])
        return tf.nn.sampled_softmax_loss(
            self.w, self.b, inputs_, labels_, self.num_samples, vocab_size)
    
      loss = tf.nn.seq2seq.sequence_loss_by_example(
            [self.output],
            [tf.reshape(self.targets, [-1,1])],
            [self.mask],
            softmax_loss_function=_loss_fct)
    
    elif fct == "nce":
      loss = tf.nn.nce_loss(self.w, self.b,                           
                            self.output,
                            tf.reshape(self.targets, [-1,1]),
                            self.num_samples, 
                            self.vocab_size)
    else:
      raise ValueError("Unsupported loss function: %s" % fct) 
    return loss, logits

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  def save_config(self):
    self.config.step = self.step
    self.config.epoch = self.epoch

    self.config.save()
  
  @property
  def initial_state(self):
    return self._initial_state

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  @property
  def data_type(self):
    return tf.float16 if self.use_fp16 else tf.float32

  @property
  def batch_size(self):
    return self.config.batch_size
  
  @property
  def vocab_size(self):
    # vocab_size is increase by two for <eos> a <bos>
    # since the first index 0 is used for padding
    return self.config.vocab_size+2
  
  @property
  def hidden_size(self):

    return self.config.hidden_size
  
  @property
  def keep_prob(self):
    return self.config.keep_prob

  @property
  def num_layers(self):
    return self.config.num_layers

  @property
  def num_samples(self):
    return self.config.num_samples

  @property
  def fast_test(self):
    return self.config.fast_test
