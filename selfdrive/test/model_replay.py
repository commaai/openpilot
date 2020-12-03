#!/usr/bin/env python3
import os
import zipfile
import tempfile
import numpy as np

from tools.lib.logreader import LogReader
from tools.lib.framereader import FrameReader
from tools.lib.cache import cache_path_for_file_path
from selfdrive.test.process_replay.camera_replay import camera_replay


if __name__ == "__main__":
  lr = LogReader(os.path.expanduser('~/rlog.bz2'))
  fr = FrameReader(os.path.expanduser('~/fcamera.hevc'))
  desire = np.load(os.path.expanduser('~/desire.npy'))
  
  try:
    msgs = camera_replay(list(lr), fr, desire=desire)
  finally:
    cache_path = cache_path_for_file_path(os.path.expanduser('~/fcamera.hecv'))
    if os.path.isfile(cache_path):
      os.remove(cache_path)

  output_path = os.path.expanduser('~/modeldata.zip')
  with zipfile.ZipFile(output_path, mode='w') as archive:
    for i, msg in enumerate(msgs):
      with tempfile.NamedTemporaryFile(mode='wb') as outfile:
        outfile.write(msg.as_builder().to_bytes())
        archive.write(outfile.name, str(i))

  print("Finished replay")
