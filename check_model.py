"""
  29 nov, 2016 - pltrdy
    
  Utility that load a model a tells whether it is optimized for softmax (w_t)
  or sampled_softmax.

  see transpose.py for more information

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import sys
# We put config in a separate file so that loading a config object does (using pickle)
# import this file twice (which triggers error)
from config import *



flags = tf.flags
logging = tf.logging

flags.DEFINE_string("model_dir", None, "model_dir (containing ckpt files and word_to_id)")
FLAGS = flags.FLAGS


def data_type():
  return tf.float32


from config import Config
def get_config():
  config_path = os.path.join(FLAGS.model_dir, "config")
  return Config(path=config_path) 

def _restore_session(saver, session):
  ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(session, ckpt.model_checkpoint_path)
    return session
  else:
    raise ValueError("No checkpoint file found") 


import os
import pickle
def main(_):
  assert(FLAGS.model_dir is not None)
  

  config = get_config()
  eval_config = Config(clone=config)
  eval_config.batch_size = 1
  eval_config.num_step = 1

  initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
  with tf.Graph().as_default():
    size = config.hidden_size
    vocab_size = config.vocab_size


    with tf.variable_scope("Model", reuse=False, initializer=initializer):

      w = tf.get_variable("w", [vocab_size, size], dtype=data_type())
      w_t = tf.get_variable("w_t", [size, vocab_size], dtype=data_type())
      
      no_w = False
      no_wt = False
      with tf.Session() as session:
        # Trying w
        o = None
        saver = tf.train.Saver([w])
        try:
          session = _restore_session(saver, session)  
          o = w
        except tf.errors.NotFoundError:
          no_w = True

        # Trying w_t
        saver = tf.train.Saver([w_t])
        try:
          session = _restore_session(saver, session)
          o = w_t
        except tf.errors.NotFoundError:
          no_wt = True

        print("\n\n===============================\n")
        print("w (opti sampledsoftmax): %s\nwt (opti softmax): %s" % (str(not no_w), str(not no_wt)))
        
        if not (no_w and no_wt):
          print("Shape: %s" % str(session.run(tf.shape(o))))
             
if __name__ == "__main__":
  tf.app.run()

