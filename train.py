#!/usr/bin/env python3
from rnnlm_ops import RnnlmOp, run_epoch
from dataset import Datasets
from config import Config
import os
import tensorflow as tf 

class Train(RnnlmOp):
  def __init__(self, config, params):
    super(Train, self).__init__(config, params) 
    
    self.io.check_dir(params.data_path)
    assert(bool(params.continue_training) == params.continue_training)

    self.data_path = params.data_path
    self.continue_training = params.continue_training
    self.loss_fct = params.loss_fct

    if not self.continue_training:
      self.config.epoch, self.config.step = 1,0 

    self._load_data()
    self._build_graph()

  def _load_data(self):
    self.data = Datasets(self.data_path, 
                        training=True, 
                        word_to_id=None, 
                        batch_size=self.config.batch_size,
                        num_steps=self.config.num_steps)
    self.io.save_w2id(self.data.word_to_id)
        
  def _build_graph(self):
    config = self.config
    config.fast_test = False
    eval_config = Config(clone=config)
    eval_config.batch_size = 1
    initializer = self.model_initializer
    with tf.name_scope("Train"):
        with tf.variable_scope("Model", reuse=False, initializer=initializer):
          self.train_model = self.Model(config=config, is_training=True, loss_fct=self.loss_fct)
        tf.summary.scalar("Training Loss", self.train_model.cost)
        tf.summary.scalar("Learning Rate", self.train_model.lr)
    
        with tf.name_scope("Valid"):
          with tf.variable_scope("Model", reuse=True, initializer=initializer):
            self.validation_model = self.Model(config=config, is_training=False, loss_fct="softmax")
          tf.summary.scalar("Validation Loss", self.validation_model.cost)
    
    with tf.name_scope("Test"):
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        self.test_model = self.Model(config=eval_config, is_training=False)

  def _run(self):
    m, mvalid, mtest = self.train_model, self.validation_model, self.test_model
    config = self.config
    data = self.data
    params = self.params
        
    init_op = tf.initialize_all_variables()
    with tf.Session() as session:
      session.run(init_op)

      print("Starting training from epoch %d using %s loss" % (config.epoch, m.loss_fct))
          
      while config.epoch <= config.max_max_epoch:
        i = config.epoch
        lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)

        print("\nEpoch: %d Learning rate: %.3f" % (i, session.run(m.lr)))
        train_perplexity = run_epoch(session, m,
          data.train,
          eval_op=m.train_op,
          verbose=True,
          opIO=self.io,
          log_rate=params.log_rate,
          save_rate=params.save_rate)
        print("Epoch: %d Train Perplexity: %.3f" % (i, train_perplexity))

        print("Validation using %s loss" % mvalid.loss_fct)
        valid_perplexity = run_epoch(session, mvalid, data.valid)
        print("Epoch: %d Valid Perplexity: %.3f" % (i, valid_perplexity))

        config.step = 0
        config.epoch += 1
        config.save()

        self.io.save_checkpoint(session, "ep_%d.ckpt" % config.epoch)

if __name__ == "__main__":
  import flags
  train = Train(flags.config, flags.FLAGS) 
  train()
