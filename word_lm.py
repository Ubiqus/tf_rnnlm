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
- (obsolete with dynamic) num_steps - the number of unrolled steps of LSTM
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

from config import *
from model import Model
import reader
from dataset import Datasets, SingleSentenceData

ACTIONS = ["test", "train", "ppl", "continue", "loglikes", "generate"]
LOSS_FCTS = ["softmax", "nce", "sampledsoftmax"]

MODEL_PARAMS_INT = [
      "max_grad_norm"
      "num_layers",
      "hidden_size",
      "max_epoch",
      "max_max_epoch",
      "batch_size", 
      "vocab_size",
      "num_samples"]
MODEL_PARAMS_FLOAT = [
      "init_scale",
      "learning_rate",
      "keep_prob",
      "lr_decay"]

MODEL_PARAMS_BOOL = [
      "fast_test",
]

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
flags.DEFINE_integer("save_rate", None, "Number of saves per epoch (default: 'log_rate' value)")

flags.DEFINE_integer("log_rate", 10, "Number of log per epoch (default: 10)")

flags.DEFINE_integer("gwords", 0, "(with --action generate) Set how many words to generate")

flags.DEFINE_integer("gline", 50, "(with --action generate) Set how many lines to generate")

flags.DEFINE_integer("gsteps", 35, "(with action --generate) Words to use for next word prediction")

flags.DEFINE_integer("gtop", 0, "(with action --generate) Show 'gtop' highest probability words")


for param in MODEL_PARAMS_INT:
  flags.DEFINE_integer(param, None, "Manually set model %s" % param)
for param in MODEL_PARAMS_FLOAT:
  flags.DEFINE_float(param, None, "Manually set model %s" % param)
for param in MODEL_PARAMS_BOOL:
  flags.DEFINE_bool(param, None, "Manually set model %s" % param)
MODEL_PARAMS = MODEL_PARAMS_INT + MODEL_PARAMS_FLOAT + MODEL_PARAMS_BOOL



FLAGS = flags.FLAGS

def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


def run_epoch(session, model, data, eval_op=None, verbose=False, 
  outputs=['ppl'], saver=None, log_rate=10, save_rate=50, state=None):
  """Runs one epoch on the given data.
     Inputs:
      - session: at tensorflow session
      - model: a model.Model object.
      - data: a data object such as dataset.SentenceSet 
          or dataset.SingleSentenceData
          i.e. which has a 'batch_iterator()' function
          which yields a tuple of (x,y), two
          [batchsize x n] numpy arrays
      - eval_op: a tensorflow operatiohn
      - verbose: a boolean that set verbosity
      - output: a list of desired outputs in 'ppl', 'll',
          'logits', 'wps', 'loss', 'state'
      - saver: a tf.Saver object
      - log_rate: int, set the number of log per epoch
      - save_rate: int, set the number of save per epoch
      - state: set the initial state
  """
  is_pos_int = lambda x: x == int(max(0, x))
  if is_pos_int(log_rate) and is_pos_int(save_rate):
    ValueError("log_rate and save_rate must be positive integer")
  
  epoch_size = ((len(data.data) // model.batch_size) - 1)
  if not epoch_size > 1:
    ValueError("Epoch_size must be higher than 0. Decrease 'batch_size'") 
  config = model.config
  costs = 0.0
  iters, totiters = 0, 0
 
  last_step = config.step if model.is_training else 0
  if last_step > 0:
    state = _load_state()
    print("Last step: %d" % last_step)
  elif state is None:
    state = session.run(model.initial_state)
   
  start_time = time.time()
  for step, (x, y) in enumerate(data.batch_iterator()):
    if last_step > step: continue

    fetches = {
      "cost": model.cost, 
      "state": model.final_state, 
      "loss": model.loss,
      "seq_len": model.seq_len
      }
    if "logits" in outputs:
      fetches["logits"] = model.logits
    if eval_op is not None:
      fetches["eval_op"] = eval_op

    feed_dict = {}
    feed_dict[model.inputs] = x
    feed_dict[model.targets] = y
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
      print("x & y shapes: "+str(x.shape)+" "+str(y.shape))
      return -99.0

    cost = vals['cost']
    state = vals['state']
    loss = vals['loss']
    costs += np.sum(loss)
    seq_len = vals['seq_len']
    iters += np.sum(seq_len)
    totiters += x.shape[0]*x.shape[1]

    ppl = np.exp(costs / iters)
    wps = iters / (time.time() - start_time)
    epoch_percent = (step * 1.0 / epoch_size) * 100
    log_step = epoch_size // (log_rate+1)
    save_step = epoch_size // (save_rate+1)
    
    if step>0 and step<epoch_size: 
      if verbose and step % log_step == 0:
        print("[Epoch %d | Step: %d/%d(%.0f%%)]" % (config.epoch,step, epoch_size, 
                                                  epoch_percent)
          +"\tTraining Perplexity: %.3f" % ppl
          +"\tSpeed: %.0f wps" % wps
          +"\tPad Ratio: %.3f" % (1-(iters/totiters)))
        sys.stdout.flush()

      if saver is not None and step % save_step == 0:
        print("[Epoch %d | Step: %d/%d(%.0f%%)]\t" % (config.epoch,step, epoch_size,
                                                   epoch_percent),end="")

        _save_checkpoint(saver, session, "ep_%d_step_%d.ckpt" % (config.epoch, step))
        _save_state(state) 
        config.step = step
        config.save() 
  # Reseting step at end of epoch
  config.step = 0
  
  # Perplexity and loglikes
  ppl = np.exp(costs / iters)
  ll = -costs / np.log(10)

  # Output dict
  out = {}
  if "ll" in outputs: out['ll'] = ll
  if "ppl" in outputs: out['ppl'] = ppl
  if "wps" in outputs: out['wps'] =  wps
  if "loss" in outputs: out['loss']= loss
  if "logits" in outputs: out["logits"] = vals["logits"]
  if "state" in outputs: out['state'] = state
  # Return directly the value if there's only one
  if len(outputs) == 1:
    return out[outputs[0]]
  return out


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
  generate = action == "generate"

  linebyline = ppl or loglikes or generate
  test = action == "test"

  log_rate = FLAGS.log_rate
  save_rate = FLAGS.save_rate
  if save_rate is None:
    save_rate = log_rate

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
   
  config.fast_test = fast_test
  
  eval_config = Config(clone=config)
  eval_config.batch_size=1

  # Load data
  if not linebyline:
    #raw_data = reader.raw_data(FLAGS.data_path, training=train, word_to_id=word_to_id)
    #train_data, valid_data, test_data, word_to_id = raw_dat
    data = Datasets(FLAGS.data_path, training=train, word_to_id=word_to_id, batch_size=config.batch_size)
    word_to_id = data.word_to_id
  
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
          m = Model(config=config, is_training=True, loss_fct=loss_fct)
        tf.scalar_summary("Training Loss", m.cost)
        tf.scalar_summary("Learning Rate", m.lr)
      
      with tf.name_scope("Valid"):
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
          mvalid = Model(config=config, is_training=False)
        tf.scalar_summary("Validation Loss", mvalid.cost)
    
    with tf.name_scope("Test"):
      with tf.variable_scope("Model", reuse=train, initializer=initializer):
        mtest = Model(config=eval_config, is_training=False, test_opti=fast_test)

   
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

          print("\nEpoch: %d Learning rate: %.3f" % (i, session.run(m.lr)))
          train_perplexity = run_epoch(session, m, 
            data.train, 
            eval_op=m.train_op,
            verbose=True, 
            saver=saver, 
            log_rate=log_rate, 
            save_rate=save_rate)
          print("Epoch: %d Train Perplexity: %.3f" % (i, train_perplexity))
          
          valid_perplexity = run_epoch(session, mvalid, data.valid)
          print("Epoch: %d Valid Perplexity: %.3f" % (i, valid_perplexity))
          
          config.step = 0
          config.epoch += 1
          config.save()

          _save_checkpoint(saver, session, "ep_%d_step_%d.ckpt" % (config.epoch, 0))
        
      else:
        session = _restore_session(saver, session)

        if loglikes:
          inputs = sys.stdin
          
          singsen = SingleSentenceData()
          while True:
            senlen = singsen.read_from_file(sys.stdin, word_to_id)
            if senlen is None:
              break
            if senlen < 2:
              print(-9999)
              continue
            
            print(senlen)
            o = run_epoch(session, mtest, singsen)
            print("ppl %.3f" % o)
  
        elif generate:
          nline = FLAGS.gline
          nsteps = FLAGS.gsteps
          top = FLAGS.gtop
          idict = dict(zip(word_to_id.values(), word_to_id.keys()))
          line = " "
          state = None
          linecount = 0
          while linecount < nline:
            singsen = SingleSentenceData()
            senlen = singsen.set_line(line, word_to_id)
            
            o = run_epoch(session, mtest, singsen, outputs=['loss', 'logits', 'state'], state=state)
            state = o['state']
            probs = o['logits'][-1]
            probs = np.exp(probs)
            probs /= sum(probs)
            
            # Print words with highest probability
            if top > 0:
              print("")
              ind = np.argpartition(probs, -top)[-top:]
              ind = ind[np.argsort(probs[ind])]
              vals= probs[ind]

              for i in range(top):
                index = ind[top-(i+1)]
                word = idict[index]
                prob = vals[top-(i+1)]
                print("#%d\t%s\t%f" % (i+1, word, prob))

            i = np.random.choice(range(len(probs)), p=probs)
            nextw = idict[i]
            
            if nextw == "<eos>":
              print("")
              line = " "
              linecount += 1
              state = None
            else:
              print(nextw, end=" ")
              sys.stdout.flush()
              line += " %s" % nextw

          # Whole text processing
        elif test:
          test_perplexity = run_epoch(session, mtest, data.test)
          print("Test Perplexity: %.3f" % test_perplexity)   
                    
if __name__ == "__main__":
  tf.app.run()

