#!/usr/bin/env python3
from __future__ import print_function, division
from rnnlm_ops import RnnlmOp, run_epoch
from dataset import SingleSentenceData
from config import Config
import sys 
from util import SpeedCounter
import tensorflow as tf

class Loglikes(RnnlmOp):
  def __init__(self, config, params):
    super(Loglikes, self).__init__(config, params)
    self._build_graph()

  def _build_graph(self):
    config = self.config
    eval_config = Config(clone=config)
    eval_config.batch_size = 1

    initializer = self.model_initializer
    with tf.name_scope("Test"):
      with tf.variable_scope("Model", reuse=False, initializer=initializer):
        self.test_model = self.Model(config=eval_config, is_training=False)

  def _run(self):
    with tf.Session() as session:
      self.io.restore_session(session)

      inputs = sys.stdin
      singsen = SingleSentenceData()
      scounter = SpeedCounter().start()
      while True:
        senlen = singsen.read_from_file(sys.stdin, self.io.w2id)
        if senlen is None:
          break
        if senlen < 2:
          print(-9999)
          continue

        o = run_epoch(session, self.test_model, singsen)
        scounter.next()
        if self.params.progress and scounter.val % 20 ==0:
          print("\rLoglikes per secs: %f" % scounter.speed, end="", file=sys.stderr)
        print("%f" % o)

if __name__ == "__main__":
  import flags
  ll = Loglikes(flags.config, flags.FLAGS)
  ll()

