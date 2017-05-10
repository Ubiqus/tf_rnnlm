from __future__ import division, print_function
from util import SpeedCounter, mkdirs
import tensorflow as tf
from dataset import Datasets, SingleSentenceData
import model
import os
import sys
import pickle
import time
import numpy as np
from config import Config

def run_epoch(session, model, data, eval_op=None, verbose=False,
  outputs=['ppl'], opIO=None, log_rate=10, save_rate=50, state=None):
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

  epoch_size = data.epoch_size
  if not epoch_size > 1:
    ValueError("Epoch_size must be higher than 0. Decrease 'batch_size'")
  config = model.config
  costs = 0.0
  iters, totiters = 0, 0

  last_step = config.step if model.is_training else 0
  if last_step > 0 and opIO is not None:
    state = opIO.load_state()
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
    if "choices" in outputs:
      fetches["choices"] = model.choices

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

      if opIO is not None and step % save_step == 0:
        print("[Epoch %d | Step: %d/%d(%.0f%%)]\t" % (config.epoch,step, epoch_size,
                                                   epoch_percent),end="")

        opIO.save_checkpoint(session, "ep_%d_step_%d.ckpt" % (config.epoch, step))
        opIO.save_state(state)
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
  if "choices" in outputs: out['choices'] = vals['choices']
  # Return directly the value if there's only one
  if len(outputs) == 1:
    return out[outputs[0]]
  return out

class OpIO:
  def __init__(self, params):
    self.params = params
    self.word_to_id, self.id_to_word = None, None
    self._saver = None
    mkdirs(params.model_dir)
  
  def check_dir(self, path):
    if path is None:
      raise ValueError("path is None")
    if not os.path.isdir(path):
      raise ValueError("path is not a valid directory")
    return True

  def get_config(self):
    params = self.params
    params = {key: params.__getattr__(key) for key in MODEL_PARAMS} 
    config_path = os.path.join(FLAGS.model_dir, "config")
    return Config(config=FLAGS.config, path=config_path, params=params)

  def save_checkpoint(self, session, filename):
    path = os.path.join(self.model_dir, filename)
    print("Saving %s" % path)
    self.saver.save(session, path)

  def restore_session(self, session):
    ckpt = tf.train.get_checkpoint_state(self.model_dir)
    if ckpt and ckpt.model_checkpoint_path:
      self.saver.restore(session, ckpt.model_checkpoint_path)
      return session
    else:
      raise ValueError("No checkpoint file found")

  def load_w2id(self):
    with open(self.w2id_path, 'rb') as f:
      self.word_to_id = pickle.load(f)
    return self.word_to_id

  def save_w2id(self, w2id=None):
    if w2id is not None:
      self.word_to_id = w2id
    
    with open(self.w2id_path, 'wb') as f:
        pickle.dump(self.w2id, f)
 
  def load_state(self,):
    with open(self.state_path, 'rb') as f:
      return pickle.load(f)

  def save_state(self, state):
    with open(self.state_path, 'wb') as f:
      pickle.dump(state, f)

  @property
  def model_dir(self):
    return self.params.model_dir

  @property
  def w2id_path(self):
    return os.path.join(self.model_dir, "word_to_id")

  @property
  def state_path(self):
    return os.path.join(self.model_dir, "state")

  @property
  def w2id(self):
    if self.word_to_id is None:
      self.load_w2id()
    return self.word_to_id
  
  @property
  def id2w(self):
    if self.id_to_word is None:
      w2id = self.w2id
      self.id_to_word = dict(zip(w2id.values(), w2id.keys()))
    return self.id_to_word
    
  @property
  def saver(self):
    if self._saver is None:
      self._saver = tf.train.Saver()
    return self._saver

class RnnlmOp(object):
  MODELS = {"default": model.Model}
  
  
  def param_default(self, param, val):
    try:
      return self.params.__getattr__(param)
    except AttributeError:
      self.params.__setattr__(param, val)
      return val

  def __init__(self, config, params):
    if not model in RnnlmOp.MODELS: 
      ValueError("Invalid model: %s" % model)
    
    self.params = params
    self.model = self.param_default("model", "default")
    self.io = OpIO(params)
    self.config = config
    self.model_initializer = tf.random_uniform_initializer(-config.init_scale, 
                                                            config.init_scale)
    print(self.config)

  def Model(self, *args, **kwargs):
    model_class = RnnlmOp.MODELS[self.model]
    return model_class(*args, **kwargs)

  def __call__(self):
    self._run()

  def _run(self):
    raise ValueError("Nothing to do")
