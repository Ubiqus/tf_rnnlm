#!/usr/bin/env python
from __future__ import print_function, division
from rnnlm_ops import RnnlmOp, run_epoch
from dataset import Datasets
from config import Config 
import tensorflow as tf

class Test(RnnlmOp):
  def __init__(self, config, params, model="default"):
    super(Test, self).__init__(config, params)
    self.io.check_dir(params.data_path)
    self.data_path = params.data_path
    
    self._load_data()
    self._build_graph()

  def _load_data(self):
    w2id = self.io.w2id
    self.data = Datasets(self.data_path, training=False, word_to_id=w2id, batch_size=1)

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
      test_perplexity = run_epoch(session, self.test_model, self.data.test)
      print("Test Perplexity: %.3f" % test_perplexity)

  @property
  def w2id(self):
    return self.io.w2id

if __name__ == "__main__":
  import flags
  test = Test(flags.config, flags.FLAGS)
  test()
