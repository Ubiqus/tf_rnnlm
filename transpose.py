#!/usr/bin/env python

"""
  29 nov, 2016 - pltrdy
  
  Loading a model and saving with transposed weights
  The point is to be able to both train and test at
  full performance which isn't straightforward since
  training with sampled softmax uses w and testing 
  with softmax uses w_t. 
  
  Using only tf.transpose to get one fro the other 
  create performance issues.

  This is a known issue:
  https://github.com/tensorflow/tensorflow/issues/5350

  In order to get best performance one must train using 
  sampled softmax, then use this script to generate model
  with transpose weights, and use the new model to test.

  Using 'transpose' will also set 'fast_test' to True in
  the config file corresponding to the new model	
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import numpy as np
import tensorflow as tf
from config import *
import os

flags = tf.flags

flags.DEFINE_string("src", None, "Source model dir (containing ckpt files and word_to_id)")
flags.DEFINE_string("dst", None, "Destination directory")
FLAGS = flags.FLAGS


def data_type():
  return tf.float32

def _save_checkpoint(saver, session, name):
  path = os.path.join(FLAGS.dst, name)
  print("Saving %s" % path)
  saver.save(session, path)

from config import Config
def get_config():
  config_path = os.path.join(FLAGS.src, "config")
  return Config(path=config_path) 

def _restore_session(saver, session):
  ckpt = tf.train.get_checkpoint_state(FLAGS.src)
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(session, ckpt.model_checkpoint_path)
    return session
  else:
    raise ValueError("No checkpoint file found") 


def main(_):
  assert(FLAGS.src is not None)
  assert(FLAGS.dst is not None)
  

  src_config = get_config()
  config = Config(clone=src_config)
  config.batch_size = 1
  config.num_step = 1
  config.path = os.path.join(FLAGS.dst, "config")
  config.fast_test = True
  
  initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
  with tf.Graph().as_default():
    """
    We're creating a model similar to word_lm.py
    """
    size = config.hidden_size
    vocab_size = config.vocab_size


    with tf.variable_scope("Model", reuse=False, initializer=initializer):
      # Inputs won't actually be used, we just set ummy values
      dummyinput = tf.ones([config.batch_size, config.num_steps], dtype=tf.int32)

      lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=1.0, state_is_tuple=True)
      
      cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)
      _initial_state = cell.zero_state(config.batch_size, data_type())

      with tf.device("/cpu:0"):
        embedding = tf.get_variable(
            "embedding", [vocab_size, size], dtype=data_type())

        inputs = tf.nn.embedding_lookup(embedding, dummyinput)

      inputs = [tf.squeeze(input_, [1])
           for input_ in tf.split(1, config.num_steps, inputs)]
      outputs, state = tf.nn.rnn(cell=cell, inputs=inputs, initial_state=cell.zero_state(config.batch_size, data_type()))

      b = tf.get_variable("b", [vocab_size], dtype=data_type())
      w = tf.get_variable("w", [vocab_size, size], dtype=data_type())
      w_t = tf.Variable(tf.transpose(w), name="w_t")

      
      with tf.Session() as session:
        # Loading everything (but w_t) from file
        v = tf.all_variables() 
        v.remove(w_t)
      
        print("loading %s"%str(v))
        saver = tf.train.Saver(v)
        session = _restore_session(saver, session)  
        
        # Only init w_t. Otherwise the model will be
        # corrupted with input
        init_op = tf.initialize_variables([ w_t ])
        session.run(init_op)

	
        # Saving everything except w
        v.remove(w) 
	v.append(w_t)
	print("saving %s"%str(v))  
        saver = tf.train.Saver(v)
        _save_checkpoint(saver, session, "wt2w.ckpt")
        
        # Copying word_to_id
        w2i_src_path = os.path.join(FLAGS.src, "word_to_id")
        w2i_dst_path = os.path.join(FLAGS.dst, "word_to_id")
        shutil.copyfile(w2i_src_path, w2i_dst_path)
        
        # Saving config to dst dir
        config.save() 

if __name__ == "__main__":
  tf.app.run()

