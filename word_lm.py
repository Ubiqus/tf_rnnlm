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

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import json
import numpy as np
import tensorflow as tf
import sys
# We put config in a separate file so that loading a config object does (using pickle)
# import this file twice (which triggers error)
from config import *

# Using our custom reader
import reader

ACTIONS = ["test", "train", "ppl", "predict", "continue", "loglikes"]
LOSS_FCTS = ["softmax", "nce", "sampledsoftmax"]

MODEL_PARAMS_INT = [
      "max_grad_norm"
      "num_layers",
      "num_steps",
      "hidden_size",
      "max_epoch",
      "max_max_epoch",
      "batch_size", 
      "vocab_size"]
MODEL_PARAMS_FLOAT = [
      "init_scale",
      "learning_rate",
      "keep_prob",
      "lr_decay"]


flags = tf.flags
logging = tf.logging

flags.DEFINE_string("model_dir", "model", "model_dir (containing ckpt files and word_to_id)")
flags.DEFINE_string("action", "test", "should we train or test. Possible options are: %s" % ", ".join(ACTIONS))
flags.DEFINE_string(
    "config", None,
    "A type of model. Possible options are: 'small', 'medium', 'large' or path to config file.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("loss", "softmax", 
                    "The loss function to use. Possible options are %s" % ", ".join(LOSS_FCTS))
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

flags.DEFINE_bool("nosave", False, "Set to force model not to be saved")
flags.DEFINE_integer("log", 10, "How often to print information and save model: each (epoch_size/log) steps. (--log 100: each 1% --log 50: each 2%, --log 10: each 10% etc")

for param in MODEL_PARAMS_INT:
  flags.DEFINE_integer(param, None, "Manually set model %s" % param)
for param in MODEL_PARAMS_FLOAT:
  flags.DEFINE_float(param, None, "Manually set model %s" % param)
MODEL_PARAMS = MODEL_PARAMS_INT + MODEL_PARAMS_FLOAT


FLAGS = flags.FLAGS


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32





class Model(object):
  """The model."""

  def __init__(self, is_training, config, loss_fct="softmax", test_opti=False):
    self.config = config
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    size = config.hidden_size
    self.vocab_size = vocab_size = config.vocab_size
    self.loss_fct = loss_fct
    self.is_training = is_training

    self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
    if is_training and config.keep_prob < 1:
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

    self._initial_state = cell.zero_state(batch_size, data_type())
    
    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, self._input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

  
    inputs = [tf.squeeze(input_, [1])
           for input_ in tf.split(1, num_steps, inputs)]
    outputs, state = tf.nn.rnn(cell=cell, inputs=inputs, initial_state=self._initial_state)
    output = tf.reshape(tf.concat(1, outputs), [-1, size])

    # We are not masking loss anyway, so it is basically only ones
    # Still may change in near future
    mask = tf.ones([self.batch_size * self.num_steps])

    if test_opti:
      # If test_opti is True we assume that the model has w_t variable 
      # this makes a huge performance improvement (especially on large model/vocab
      # but require model weights to be transposed.
      # See transpose.py 
      self.w_t = w_t = tf.get_variable("w_t", [size, vocab_size], dtype=data_type())
      self.b = b = tf.get_variable("b", [vocab_size], dtype=data_type())
      loss, logits = self.softmax(output, w_t, b, mask)
      self.logits = logits

    elif not is_training or self.loss_fct == "softmax": 
      # Regular testing using softmax and default "w" weights matrix
      # It may be really slower than 'test_opti'
      w = tf.get_variable("w", [vocab_size, size], dtype=data_type())
      b = tf.get_variable("b", [vocab_size], dtype=data_type()) 
      w_t = tf.transpose(w)
      
      loss, logits = self.softmax(output, w_t, b, mask)
      self.logits = logits

    elif self.loss_fct == "nce":
      w = tf.get_variable("w", [vocab_size, size], dtype=data_type())
      b = tf.get_variable("b", [vocab_size], dtype=data_type())
      num_samples = 64
      labels = tf.reshape(self._targets, [-1,1])
      hidden = output
      loss = tf.nn.nce_loss(w, b,                           
                            hidden,
                            labels,
                            num_samples, 
                            vocab_size)
    elif self.loss_fct == "sampledsoftmax":
      w = tf.get_variable("w", [vocab_size, size], dtype=data_type())
      b = tf.get_variable("b", [vocab_size], dtype=data_type())
      num_samples = 1024
      labels = tf.reshape(self._targets, [-1,1])
      hidden = output
      
      loss = self.sampled_softmax(w, b, labels, hidden, num_samples, mask)

    else:
      raise ValueError("Unsupported loss function: %s" % loss_fct) 
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state 
    self.probs = loss
    
    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  def sampled_softmax(self, w, b, labels, hidden, num_samples, mask):
    vocab_size = self.vocab_size
    def _loss_fct(inputs_, labels_):
        labels_ = tf.reshape(labels_, [-1, 1])
        return tf.nn.sampled_softmax_loss(
            w, b, inputs_, labels_, num_samples, vocab_size)
    
    loss = tf.nn.seq2seq.sequence_loss_by_example(
        [ hidden],
        [ labels ],
        [ mask],
        num_samples,
        softmax_loss_function=_loss_fct)
    return loss

  def softmax(self, output, w, b, mask):
    logits = tf.matmul(output, w)+b
    loss = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(self._targets, [-1])],
        [mask])
    return loss, logits

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


def run_epoch(session, model, data, eval_op=None, verbose=False, idict=None, saver=None, loglikes=False, log=10):
  """Runs the model on the given data.
      Returns:
        - if idict is set (prediction mode):
            a tuple ppl, predictions
        - else if loglikes:
            loglikes (-costs/log(10))
        - else:
            perxplexity= exp(costs/iters)
  """
  epoch_size = ((len(data) // model.batch_size) - 1) // model.num_steps
  config = model.config
  costs = 0.0
  iters = 0 
 
  last_step = config.step if model.is_training else 0
  if last_step > 0:
    state = _load_state()
    print("Last step: %d" % last_step)
  else: 
    state = session.run(model.initial_state)
  predictions = []
  
  start_time = time.time()
  for step, (x, y) in enumerate(reader.iterator(data, model.batch_size,
                                                    model.num_steps)):
    if last_step > step: continue

    fetches = {"cost": model.cost, "state": model.final_state, "probs": model.probs}

    if eval_op is not None:
      fetches["eval_op"] = eval_op

    feed_dict = {}
    feed_dict[model._input_data] = x
    feed_dict[model._targets] = y
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h
   

    # Catching error & returning -99 as we may need an output for each input
    # (can't just ignore)
    try: 
      vals = session.run(fetches, feed_dict)
    except ValueError as e:
      print("[ERROR] Error while running step %d (value: =\"%s\")" % (step, str(x)),
		file=sys.stderr)
      print("[ERROR] Aborting run_step; returning -99", file=sys.stderr)
      print(e, file=sys.stderr)
      return -99.0

    cost = vals['cost']
    state = vals['state']
    probs = vals['probs']
    costs += cost
    iters += model.num_steps
    
    # Predict mode
    if idict is not None:
      probs = probs[0]
      next_id = np.argmax(probs)
      xx = x[0][0]
      yy = y[0][0]
      prediction = {
        'word': idict[xx], 
        "target": idict[yy], 
        "prob": float(probs[yy]), 
        "pred_word": idict[next_id],
        "pred_prob": float(probs[next_id]),
        }
      predictions.append(prediction)

    # Logging results & Saving
    if log<0 or log>100:
      log = 10
    if verbose and step % (epoch_size // log) == 1:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters * model.batch_size / (time.time() - start_time)))
      if saver is not None:
        _save_checkpoint(saver, session, "ep_%d_step_%d.ckpt" % (config.epoch, step))
        _save_state(state) 
        config.step = step
        config.save() 
  

  config.step = 0

  
  # Perplexity and loglikes
  ppl = np.exp(costs / iters)
  ll = -costs / np.log(10)
  if not idict:
    ret = ll if loglikes else ppl  
    return ret
  return ppl, predictions

def _save_checkpoint(saver, session, name):
  path = os.path.join(FLAGS.model_dir, name)
  print("Saving %s" % path)
  saver.save(session, path)

def _state_path():
  return os.path.join(FLAGS.model_dir, "state")

def _load_state():
  with open(_state_path(), 'r') as f:
    return pickle.load(f)

def _save_state(state):
  with open(_state_path(), 'w') as f:
    pickle.dump(state, f)  

from config import Config
def get_config():
  params = {key: FLAGS.__getattr__(key) for key in MODEL_PARAMS} 
  config_path = os.path.join(FLAGS.model_dir, "config")
  return Config(config=FLAGS.config, path=config_path, params=params) 

def _restore_session(saver, session):
  ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(session, ckpt.model_checkpoint_path)
    return session
  else:
    raise ValueError("No checkpoint file found") 


import os
import pickle
import util
def main(_):
  assert(FLAGS.action in ACTIONS)
  assert(FLAGS.loss in LOSS_FCTS)
  
  loss_fct = FLAGS.loss
  action = FLAGS.action

  train = action in ["train", "continue"]
  ppl = action == "ppl"
  loglikes = action == "loglikes"
  predict = action == "predict"
  linebyline = ppl or loglikes or predict
  test = action == "test"

  util.mkdirs(FLAGS.model_dir)

  if not (FLAGS.data_path or linebyline):
    raise ValueError("Must set --data_path to data directory")

  config = get_config()

  word_to_id_path = os.path.join(FLAGS.model_dir, "word_to_id")
  if action != "train":
    with open(word_to_id_path, 'r') as f:
      word_to_id = pickle.load(f)

  else:
    word_to_id = None
    config.epoch = 1
    config.step = 0

  # Reading fast_test. 
  # This option is enabled by 'transpose.py'
  fast_test = False
  if "fast_test" in config.__dict__:
    # Be sure to set a boolean
    fast_test = True if config.fast_test else False

  # Warning if we're slowing testing
  if not (fast_test or train):
    print("""\n\n[WARNING]: You are using a test feature involving 'softmax'
               you must consider using 'fast_test' feature'""")
    print("[WARNING]: See transpose.py for more information")
   
  eval_config = Config(clone=config)

  # Load data
  if not linebyline:
    raw_data = reader.raw_data(FLAGS.data_path, training=train, word_to_id=word_to_id)
    train_data, valid_data, test_data, word_to_id = raw_data
  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    # Defining model(s)
    if train:
      # Saving word_to_id & conf to file
      with open(word_to_id_path, 'w') as f:
        pickle.dump(word_to_id, f)

      with tf.name_scope("Train"):
        with tf.variable_scope("Model", reuse=False, initializer=initializer):
          m = Model(is_training=True, config=config, loss_fct=loss_fct)
        tf.scalar_summary("Training Loss", m.cost)
        tf.scalar_summary("Learning Rate", m.lr)
      
      with tf.name_scope("Valid"):
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
          mvalid = Model(is_training=False, config=config)
        tf.scalar_summary("Validation Loss", mvalid.cost)
    
    with tf.name_scope("Test"):
      with tf.variable_scope("Model", reuse=train, initializer=initializer):
        mtest = Model(is_training=False, config=eval_config, test_opti=fast_test)

   
    saver = tf.train.Saver()
    init_op = tf.initialize_all_variables()
    with tf.Session() as session:
      session.run(init_op)
      if train:
        config.save()
        if action == "continue":
          session = _restore_session(saver, session)
       
        saver = None if FLAGS.nosave else saver 
        print("Starting training from epoch %d using %s" % (config.epoch, loss_fct))
        
        while config.epoch <= config.max_max_epoch:
          i = config.epoch
          lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
          m.assign_lr(session, config.learning_rate * lr_decay)

          print("Epoch: %d Learning rate: %.3f" % (i, session.run(m.lr)))
          train_perplexity = run_epoch(session, m, train_data, eval_op=m.train_op,
                                       verbose=True, saver=saver, log=FLAGS.log)
          print("Epoch: %d Train Perplexity: %.3f" % (i, train_perplexity))
          
          valid_perplexity = run_epoch(session, mvalid, valid_data)
          print("Epoch: %d Valid Perplexity: %.3f" % (i, valid_perplexity))
          
          config.step = 0
          config.epoch += 1
          config.save()
        
      else:
        session = _restore_session(saver, session)

        # Line by line processing (=ppl, predict, loglikes)
        if linebyline:
          if predict: print("[")
          while True:
            line = sys.stdin.readline()
            if not line: break
            
            idict = None
            d = line.decode("utf-8").replace("\n", " <eos> ").split()
            test_data = [word_to_id[word] for word in d if word in word_to_id]
          
            if len(test_data) < 2:
              print(-9999)
              continue

            # Prediction mode
            if predict:
              inverse_dict = dict(zip(word_to_id.values(), word_to_id.keys()))
              ppl, predict = run_epoch(session, mtest, test_data, idict=inverse_dict)
              res = {'ppl': ppl, 'predictions': predict}
              print(json.dumps(res)+",")
            
            # ppl or loglikes
            else:
              o = run_epoch(session, mtest, test_data, loglikes=loglikes)
              print("%.3f" % o)
              

          if predict: print("]")

          # Whole text processing
        elif test:
          test_perplexity = run_epoch(session, mtest, test_data)
          print("Test Perplexity: %.3f" % test_perplexity)   
                    
if __name__ == "__main__":
  tf.app.run()

