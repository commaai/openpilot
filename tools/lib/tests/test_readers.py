import pytest
import requests
import tempfile

from collections import defaultdict
import numpy as np
from openpilot.tools.lib.framereader import FrameReader
from openpilot.tools.lib.logreader import LogReader


class TestReaders:
  @pytest.mark.skip("skip for bandwidth reasons")
  def test_logreader(self):
    def _check_data(lr):
      hist = defaultdict(int)
      for l in lr:
        hist[l.which()] += 1

      assert hist['carControl'] == 6000
      assert hist['logMessage'] == 6857

    with tempfile.NamedTemporaryFile(suffix=".bz2") as fp:
      r = requests.get("https://github.com/commaai/comma2k19/blob/master/Example_1/b0c9d2329ad1606b%7C2018-08-02--08-34-47/40/raw_log.bz2?raw=true", timeout=10)
      fp.write(r.content)
      fp.flush()

      lr_file = LogReader(fp.name)
      _check_data(lr_file)

    lr_url = LogReader("https://github.com/commaai/comma2k19/blob/master/Example_1/b0c9d2329ad1606b%7C2018-08-02--08-34-47/40/raw_log.bz2?raw=true")
    _check_data(lr_url)

  @pytest.mark.skip("skip for bandwidth reasons")
  def test_framereader(self):
    def _check_data(f):
      assert f.frame_count == 1200
      assert f.w == 1164
      assert f.h == 874
      
      first_30 = []
      for fidx in range(0, 30):
        first_30.append(f.get(fidx))
        

      frame_0 = f.get(0,)
      frame_15 = f.get(15)

      assert np.all(first_30[0] == frame_0[0])
      assert np.all(first_30[15] == frame_15[0])

    with tempfile.NamedTemporaryFile(suffix=".hevc") as fp:
      r = requests.get("https://github.com/commaai/comma2k19/blob/master/Example_1/b0c9d2329ad1606b%7C2018-08-02--08-34-47/40/video.hevc?raw=true", timeout=10)
      fp.write(r.content)
      fp.flush()

      fr_file = FrameReader(fp.name)
      _check_data(fr_file)

    fr_url = FrameReader("https://github.com/commaai/comma2k19/blob/master/Example_1/b0c9d2329ad1606b%7C2018-08-02--08-34-47/40/video.hevc?raw=true")
    _check_data(fr_url)
