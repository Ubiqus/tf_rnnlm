import tensorflow as tf
import os

LOSS_FCTS = ["softmax", "nce", "sampledsoftmax"]



flags = tf.flags
logging = tf.logging

flags.DEFINE_bool("continue", False, "Continue training where it stopped")
flags.DEFINE_string("model_dir", "model", "model_dir (containing ckpt files and word_to_id)")
flags.DEFINE_string(
    "config", None,
    "A type of model. Possible options are: 'small', 'medium', 'large'")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("loss", "softmax", 
                    "The loss function to use. Possible options are %s" % ", ".join(LOSS_FCTS))
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

flags.DEFINE_bool("nosave", False, "Set to force model not to be saved")

flags.DEFINE_integer("save_rate", 0, "Number of saves per epoch (default: 0)")
flags.DEFINE_integer("log_rate", 10, "Number of log per epoch (default: 10)")

flags.DEFINE_integer("gline", 50, "(with --action generate) Set how many lines to generate")

flags.DEFINE_bool("progress", False, "Print progress info on stderr")

MODEL_PARAMS_INT = [
      "max_grad_norm"
      "num_layers",
      "hidden_size",
      "max_epoch",
      "max_max_epoch",
      "batch_size", 
      "vocab_size",
      "num_steps",
      "num_samples",
      "embed_dim"]
MODEL_PARAMS_FLOAT = [
      "init_scale",
      "learning_rate",
      "keep_prob",
      "lr_decay"]

MODEL_PARAMS_BOOL = [
      "fast_test",
]
for param in MODEL_PARAMS_INT:
  flags.DEFINE_integer(param, None, "Manually set model %s" % param)
for param in MODEL_PARAMS_FLOAT:
  flags.DEFINE_float(param, None, "Manually set model %s" % param)
for param in MODEL_PARAMS_BOOL:
  flags.DEFINE_bool(param, None, "Manually set model %s" % param)
MODEL_PARAMS = MODEL_PARAMS_INT + MODEL_PARAMS_FLOAT + MODEL_PARAMS_BOOL

FLAGS = flags.FLAGS

from config import Config
def get_config():
  params = {key: FLAGS.__getattr__(key) for key in MODEL_PARAMS} 
  config_path = os.path.join(FLAGS.model_dir, "config")
  return Config(config=FLAGS.config, path=config_path, params=params) 

config = get_config()
FLAGS.continue_training = FLAGS.__getattr__('continue')
FLAGS.loss_fct = FLAGS.loss
