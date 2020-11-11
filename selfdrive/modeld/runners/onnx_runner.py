#!/usr/bin/env python3
# TODO: why are the keras models saved with python 2?
from __future__ import print_function

import os
import sys
import numpy as np
import onnxruntime as ort

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
  ishapes = [[1]+ii.shape[1:] for ii in m.get_inputs()]
  keys = [x.name for x in m.get_inputs()]
  print("ready to run onnx model", keys, ishapes, file=sys.stderr)
  while 1:
    inputs = []
    for shp in ishapes:
      ts = np.product(shp)
      #print("reshaping %s with offset %d" % (str(shp), offset), file=sys.stderr)
      inputs.append(read(ts).reshape(shp))
    ret = m.run(None, dict(zip(keys, inputs)))
    #print(ret, file=sys.stderr)
    for r in ret:
      write(r)


if __name__ == "__main__":
  ort_session = ort.InferenceSession(sys.argv[1])
  run_loop(ort_session)

