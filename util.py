from __future__ import division
import os 
import errno
import time

def mkdirs(dirname):
  """Recurcively (and safely) create dir
     It is equivalent to mkdir -p dirname
  """

  try:
    os.makedirs(dirname)
  except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise exc

def fltohm(seconds):
  seconds = float(seconds)
  m, s = divmod(seconds, 60)
  h, m = divmod(m, 60)
  return "%d:%02d:%02d" % (h, m, s)

class SpeedCounter:
  def __init__(self, val=0, step=1, tot=None):
    self.val = val
    self.step = step
    self.stime = None
    self.tot = tot

  def start(self):
    if self.stime is None:
      self.stime = time.time()
    else:
      raise ValueError("SpeedCounter already started")
    return self

  def reset(self):
    self.stime = time.time()

  def next(self):
    self.val += self.step

  def steps(self, n):
    self.val += n * self.step

  def add(self, val):
    self.val += val

  @property
  def speed(self):
    return self.val / self.elapsed_time
  
  @property
  def etime(self): return self.elapsed_time
  
  @property
  def elapsed_time(self):
    return time.time() - self.stime

  @property
  def str_rtime(self):
    return fltohm(self.rtime)
  @property
  def rtime(self): return self.remaining_time

  @property
  def progress(self):
    if self.tot is not None:  
      return self.val / self.tot
    else:
      raise ValueError("Can't progress  without 'tot'")

  @property
  def remaining_time(self):
    ratio = self.progress
    etime = self.etime
    return etime * (1/self.progress)-etime
