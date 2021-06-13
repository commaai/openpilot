#!/usr/bin/env python3
import os
import numpy as np

from tools.lib.logreader import LogReader
from tools.lib.framereader import FrameReader
from tools.lib.cache import cache_path_for_file_path
from selfdrive.test.process_replay.model_replay import model_replay


if __name__ == "__main__":
  lr = LogReader(os.path.expanduser('~/rlog.bz2'))
  fr = FrameReader(os.path.expanduser('~/fcamera.hevc'))
  desire = np.load(os.path.expanduser('~/desire.npy'))
  calib = np.load(os.path.expanduser('~/calib.npy'))

  try:
    msgs = model_replay(list(lr), fr, desire=desire, calib=calib)
  finally:
    cache_path = cache_path_for_file_path(os.path.expanduser('~/fcamera.hevc'))
    if os.path.isfile(cache_path):
      os.remove(cache_path)

  output_size = len(np.frombuffer(msgs[0].model.rawPredictions, dtype=np.float32))
  output_data = np.zeros((len(msgs), output_size), dtype=np.float32)
  for i, msg in enumerate(msgs):
    output_data[i] = np.frombuffer(msg.model.rawPredictions, dtype=np.float32)
  np.save(os.path.expanduser('~/modeldata.npy'), output_data)

  print("Finished replay")
