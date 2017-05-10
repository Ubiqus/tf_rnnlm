"""

  December 2016 - pltrdy
  Handling datasets

  Notes: we assume that <eos>'s id is 1.

"""
import numpy as np
import os
import tensorflow as tf
from collections import Counter

# Don't change those values. I would have serious side effects on the model. 
EOS = "<eos>"
IEOS=2

BOS = "<bos>"
IBOS=1

PAD = "<pad>"
IPAD=0


# Cut sentences after MAX_LEN words
MAX_LEN = 100

class SingleSentenceData:
  """
    A class meant to work on a Single Sentence.
    i.e. reading a file `input_fd` line by line
    
    The point is to run a (tiny) epoch on each line
    thus each batch contains only 1 sentence.
    One iterate over the lines by using `next()`
  """
  def __init__(self):
    self.batch_size = 1  
    self.sentence = None
 
  
  def set_line(self, line, word_to_id):
    words = (" <bos> "+(line.decode("utf-8"))).replace("\n", " <eos> ").split()+['<eos>']

    sentence = [word_to_id[word] for word in words if word in word_to_id]
    self.sentence = sentence
    return len(sentence)

  def read_from_file(self, input_fd, word_to_id):
    line = input_fd.readline()
    if  not line:
      return None 
    return self.set_line(line, word_to_id)
    
  def batch_iterator(self):
    sen_len = len(self.sentence)
    x = np.zeros([1, sen_len-1]) + self.sentence[:-1]
    y = np.zeros([1, sen_len-1]) + self.sentence[1:]
    yield (x, y)

  @property
  def data(self):
    return self.sentence

  @property
  def epoch_size(self):
    return ((len(self.data) // self.batch_size) - 1)

class SentenceSet:
  """
    A class defining sentence based sub-dataset
    i.e. train, valid & text sets
    each element is a sentence
  """
  def __init__(self, raw, batch_size, shuffle=True):
    self.sentences = self._raw_to_sentences(raw)
    self.batch_size = batch_size
    self.shuffle = shuffle 
    # nb sentence in data
    self.n_sentences = len(self.data)

    # n_iter aka. 'epoch_size'
    self.n_iter = self.n_sentences // self.batch_size

    self.sentences.sort(key=len)
    self.order = np.arange(self.n_iter)

    print(self)
    
  def __str__(self):
    return ("SentenceSet:\n"  
           + "\n\t* #sentences: %d" % self.n_sentences
           + "\n\t* batch_size: %d" % self.batch_size
           + "\n\t* #iter: %d" % self.n_iter
           + "\n\t* shuffled: %s" % str(self.shuffle))

  def maybe_shuffle(self):
    if self.shuffle:
      np.random.shuffle(self.order)

  def _raw_to_sentences(self, raw_data):
    """ 
      Inputs:
        * raw_data: list of word indentifier.  [ int ]. 
      Output: 
        * sentences
    """
    ieos = IEOS
    sentences = []

    sentence = []
    count = 0
    for d in raw_data:
      sentence.append(d)
      if d == ieos:
        sentences.append(sentence)
        sentence = []

    return sentences

  def batch_iterator(self):
    n_iter, batch_size = self.n_iter, self.batch_size
    if n_iter == 0:
      raise ValueError("epoch_size == 0, decrease batch_size")
   
    self.maybe_shuffle()

    # Batching data
    for i in range(n_iter):
      ii = self.order[i]
      batch_sentences = self.sentences[batch_size*ii:batch_size*(ii+1)]
      max_len = min(max([len(s) for s in batch_sentences]), MAX_LEN)
      x = np.zeros([batch_size, max_len])+IPAD
      y = np.zeros([batch_size, max_len])+IPAD
    
      for j in range(batch_size):
        s = batch_sentences[j] 
        l = min(80, len(s))
        x[j][:l] = [IBOS]+s[:l-1]
        y[j][:l] = s[:l]

      yield (x, y)
  
  
  @property
  def data(self):
    return self.sentences

  @property
  def epoch_size(self):
    return ((len(self.data) // self.batch_size) - 1)

class SequenceSet:
  """
    Set of fixed size sequences (num_steps)
  """
  def __init__(self, raw_data, batch_size, num_steps):
    self.raw_data = np.array(raw_data, dtype=np.int32)
    self.num_steps = num_steps 
    self.batch_size = batch_size
    
    self.data_len = data_len = len(raw_data)
    self.batch_len = batch_len = data_len // batch_size
    self.epoch_size = (batch_len - 1) // num_steps
    
    print(self)

  def __str__(self):
    return ("SequenceSet:"
            + "\n\t* #sequences: %d" % self.data_len
            + "\n\t* batch_size: %d" % self.batch_size
            + "\n\t* batch_len: %d" % self.batch_len
            + "\n\t* #iter: %d" % self.epoch_size)
  
  def batch_iterator(self):
    """Iterate on the raw data.
    Args:
      raw_data: one of the raw data outputs from ptb_raw_data.
      batch_size: int, the batch size.
      num_steps: int, the number of unrolls.
    Yields:
      Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
      The second element of the tuple is the same data time-shifted to the
      right by one.
    Raises:
      ValueError: if batch_size or num_steps are too high.
    """
    # PTB Iterator from tensorflow.models.rnn.ptb.reader.py
    # on TensorFlow 0.11
    # https://github.com/tensorflow/tensorflow/blob/282823b877f173e6a33bbc9d4b9ad7dd8413ada6/tensorflow/models/rnn/ptb/reader.py
   
    raw_data = self.raw_data
    num_steps = self.num_steps
    batch_size = self.batch_size
    batch_len = self.batch_len
    epoch_size = self.epoch_size

    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
      data[i] = raw_data[batch_len * i:batch_len * (i + 1)]


    if epoch_size == 0:
      raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
      x = data[:, i*num_steps:(i+1)*num_steps]
      y = data[:, i*num_steps+1:(i+1)*num_steps+1]
      yield (x, y) 

    
class Datasets:
  """
    Managing datasets
    It may actually contains 3 datasets, namely 
    train, valid and test.
  """
  def __init__(self, path, batch_size=1, training=True, num_steps=1, word_to_id=None):
    if not training and word_to_id is None:
      raise ValueError("Must set 'word_to_id' when action is not 'train'")

    # Setting parameters
    self.path = path
    self.training = training
    self.batch_size = batch_size
    self.num_steps = num_steps
    
    # Loading from files
    train_path = os.path.join(path, "train.txt")
    valid_path = os.path.join(path, "valid.txt")
    test_path = os.path.join(path, "test.txt")
    
    if word_to_id is None:
      print("Building vocabulary...")
      self._build_vocab(train_path)
    else:
      self.word_to_id = word_to_id
    print("Vocabulary size: %d" % len(self.word_to_id))

    if training:
      print("Loading train set")
      self.train = self._load_set(train_path)
    
      print("Loading valid set")
      self.valid = self._load_set(valid_path)
    
    print("Loading test  set")
    self.test  = self._load_set(test_path, batch_size=1)

  def _load_set(self, path, batch_size=None):
    if not os.path.isfile(path):
      print("WARNING: File not found: '%s'" % path)
      return None
    
    if batch_size is None:
      batch_size = self.batch_size

    data = self._file_to_word_ids(path)

    if self.num_steps == 0:
      return SentenceSet(data, batch_size)
    else:
      return SequenceSet(data, batch_size, self.num_steps)
   
  def _build_vocab(self, filename):
    counts = Counter()
    with tf.gfile.GFile(filename, "r") as f:
      #for line in f:
      #  words = line.replace("\n"," ").split()
      #  counts += Counter(words)
      while True:
        chunk = f.read(int(500000000/2))
        if not chunk: 
          break
        counts += Counter(chunk.replace("\n", " ").split())

    sorted_pairs = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    self.word_to_id = {e[0]: (i+3) for (i, e) in enumerate(sorted_pairs)}
    self.word_to_id[EOS] = IEOS
    self.word_to_id[BOS] = IBOS
    self.word_to_id[PAD] = IPAD


  def _file_to_word_ids(self, filename):
    d = []
    w2id = self.word_to_id
    with tf.gfile.GFile(filename, "r") as f:
      for line in f:
        ids = [w2id[w] for w in line.replace("\n"," %s " % EOS).split() if w in w2id]
        d += ids

    return d
 
  def train_data(self):
    return self.train.data

  @property
  def valid_data(self):
    return self.valid.data 

  @property
  def test_data(self):
    return self.test.data
