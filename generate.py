#!/usr/bin/env python
from __future__ import print_function, division
from rnnlm_ops import RnnlmOp, run_epoch
from dataset import SingleSentenceData
from config import Config 
from util import SpeedCounter
import tensorflow as tf
import sys

class Generate(RnnlmOp):
  def __init__(self, config, params, model="default"):
    super(Generate, self).__init__(config, params)
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

      nline = self.params.gline
      w2id = self.io.w2id
      id2w = self.io.id2w
      line = " "
      state = None

      wc = 0
      lc = SpeedCounter(tot=nline).start()
      singsen = SingleSentenceData()
      while lc.val < nline:
        senlens = singsen.set_line(line, w2id)
        o = run_epoch(session, self.test_model, singsen, outputs=['state', 'choices'], state=state)
        state = o['state']
        choice = o['choices'][-1][0]
        #print(choice)
        try:
          nextw = id2w[choice]
        except KeyError:
          print("KeyError: in id2w, id=%d, ignoring" % choice, file=sys.stderr)
          continue
        if nextw != "<eos>":
          line += "%s " % nextw
          wc += 1
        else:
          lc.next()
          print(line)
          line = " "
        # informations
        if sys.stderr.isatty() and lc.val>0 and wc % 20 == 0:
          print("\rLines: %d (%2.3f%%)     Words: %d    W/L: %f    Line/sec: %.3f     Remaining time: %s" % (lc.val, 100*lc.progress, wc, wc/lc.val, lc.speed, lc.str_rtime), file=sys.stderr, end="")


if __name__ == "__main__":
  import flags
  gen = Generate(flags.config, flags.FLAGS)
  gen()
