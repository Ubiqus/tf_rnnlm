import os 
import errno

def mkdirs(dirname):
  """Recurcively (and safely) create dir
     It is equivalent to mkdir -p dirname
  """

  try:
    os.makedirs(dirname)
  except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise exc
    pass
