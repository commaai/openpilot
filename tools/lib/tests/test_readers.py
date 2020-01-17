#!/usr/bin/env python
import unittest
import requests
import tempfile

from collections import defaultdict
import numpy as np
from tools.lib.framereader import FrameReader
from tools.lib.logreader import LogReader

class TestReaders(unittest.TestCase):
  def test_logreader(self):
    with tempfile.NamedTemporaryFile(suffix=".bz2") as fp:
      r = requests.get("https://github.com/commaai/comma2k19/blob/master/Example_1/b0c9d2329ad1606b%7C2018-08-02--08-34-47/40/raw_log.bz2?raw=true")
      fp.write(r.content)
      fp.flush()

      lr = LogReader(fp.name)
      hist = defaultdict(int)
      for l in lr:
        hist[l.which()] += 1

      self.assertEqual(hist['carControl'], 6000)
      self.assertEqual(hist['logMessage'], 6857)

  def test_framereader(self):
    with tempfile.NamedTemporaryFile(suffix=".hevc") as fp:
      r = requests.get("https://github.com/commaai/comma2k19/blob/master/Example_1/b0c9d2329ad1606b%7C2018-08-02--08-34-47/40/video.hevc?raw=true")
      fp.write(r.content)
      fp.flush()

      f = FrameReader(fp.name)

      self.assertEqual(f.frame_count, 1200)
      self.assertEqual(f.w, 1164)
      self.assertEqual(f.h, 874)


      frame_first_30 = f.get(0, 30)
      self.assertEqual(len(frame_first_30), 30)


      print(frame_first_30[15])

      print("frame_0")
      frame_0 = f.get(0, 1)
      frame_15 = f.get(15, 1)

      print(frame_15[0])

    assert np.all(frame_first_30[0] == frame_0[0])
    assert np.all(frame_first_30[15] == frame_15[0])

if __name__ == "__main__":
  unittest.main()

