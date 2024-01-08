import unittest

from openpilot.tools.lib.srreader import SegmentRangeReader


class TestSegmentRangeReader(unittest.TestCase):
  def test_logreader(self):
    sr = SegmentRangeReader("a2a0ccea32023010|2023-07-27--13-01-19/0")

    for m in sr:
      print(m.which())

if __name__ == "__main__":
  unittest.main()
