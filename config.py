import json

class Config:
  """Configuration object loaded from/saved to JSON object
  """
  def __init__(self, model=None, path=None):
    self.path = path
    if model is not None:
      entries = self._get_config(model)
    elif path is not None:
      entries = self._load()
    else:
      raise ValueError("Parameters 'model' and 'path' can't be both 'None'")

    self.__dict__.update(entries)

  def _get_config(self, model):       
    if model == "small":
      o = small_config()
    elif model == "medium":
      o = medium_config()
    elif model == "large":
      o = large_config()
    elif model == "test":
      o = test_config()
    else:
      raise ValueError("Invalid model: %s", model)
    return o

  def _load(self):
    return json.load(open(self.path))

  def save(self):
    tmp = self.path
    o = self.__dict__
    del o['path']
    json.dump(o, open(tmp, 'w'), indent=2)
    self.path = tmp


def small_config():
  """Small config."""
  return {"init_scale" : 0.1,
    "learning_rate" : 1.0,
    "max_grad_norm" : 5,
    "num_layers" : 2,
    "num_steps" : 20,
    "hidden_size" : 200,
    "max_epoch" : 4,
    "max_max_epoch" : 13,
    "keep_prob" : 1.0,
    "lr_decay" : 0.5,
    "batch_size" : 20,
    "vocab_size" : 10000
    }


def medium_config():
  """Medium config."""
  return{ 
      "init_scale" : 0.05,
      "learning_rate" : 1.0,
      "max_grad_norm" : 5,
      "num_layers" : 2,
      "num_steps" : 35,
      "hidden_size" : 650,
      "max_epoch" : 6,
      "max_max_epoch" : 39,
      "keep_prob" : 0.5,
      "lr_decay" : 0.8,
      "batch_size" : 20,
      "vocab_size" : 10000,
    }

def large_config():
  """Large config."""
  return {
      "init_scale" : 0.04,
      "learning_rate" : 1.0,
      "max_grad_norm" : 10,
      "num_layers" : 2,
      "num_steps" : 35,
      "hidden_size" : 1500,
      "max_epoch" : 14,
      "max_max_epoch" : 55,
      "keep_prob" : 0.35,
      "lr_decay" : 1 / 1.15,
      "batch_size" : 20,
      "vocab_size" : 10000,
    }


def test_config():
  """Tiny config, for testing."""
  return {
    "init_scale" : 0.1,
    "learning_rate" : 1.0,
    "max_grad_norm" : 1,
    "num_layers" : 1,
    "num_steps" : 2,
    "hidden_size" : 2,
    "max_epoch" : 1,
    "max_max_epoch" : 1,
    "keep_prob" : 1.0,
    "lr_decay" : 0.5,
    "batch_size" : 20,
    "vocab_size" : 10000,
    }
