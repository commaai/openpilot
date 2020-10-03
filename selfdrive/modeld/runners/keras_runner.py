#!/usr/bin/env python3
# TODO: why are the keras models saved with python 2?
from __future__ import print_function

import tensorflow as tf  # pylint: disable=import-error
import os
import sys
import numpy as np
from tensorflow.keras.models import load_model  # pylint: disable=import-error

def read(sz):
  dd = []
  gt = 0
  while gt < sz * 4:
    st = os.read(0, sz * 4 - gt)
    assert(len(st) > 0)
    dd.append(st)
    gt += len(st)
  return np.frombuffer(b''.join(dd), dtype=np.float32)

def write(d):
  os.write(1, d.tobytes())

def run_loop(m):
  ishapes = [[1]+ii.shape[1:] for ii in m.inputs]
  print("ready to run keras model", ishapes, file=sys.stderr)
  while 1:
    inputs = []
    for shp in ishapes:
      ts = np.product(shp)
      #print("reshaping %s with offset %d" % (str(shp), offset), file=sys.stderr)
      inputs.append(read(ts).reshape(shp))
    ret = m.predict_on_batch(inputs)
    #print(ret, file=sys.stderr)
    for r in ret:
      write(r)


if __name__ == "__main__":
  print(tf.__version__, file=sys.stderr)
  # limit gram alloc
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if len(gpus) > 0:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

  m = load_model(sys.argv[1])
  print(m, file=sys.stderr)

  run_loop(m)
