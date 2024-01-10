import numpy as np
import unittest
from parameterized import parameterized

from openpilot.tools.lib.route import SegmentRange
from openpilot.tools.lib.srreader import ReadMode, SegmentRangeReader, parse_slice

NUM_SEGS = 17 # number of segments in the test route
ALL_SEGS = list(np.arange(NUM_SEGS))
TEST_ROUTE = "344c5c15b34f2d8a/2024-01-03--09-37-12"

class TestSegmentRangeReader(unittest.TestCase):
  @parameterized.expand([
    (f"{TEST_ROUTE}", ALL_SEGS),
    (f"{TEST_ROUTE.replace('/', '|')}", ALL_SEGS),
    (f"{TEST_ROUTE}--0", [0]),
    (f"{TEST_ROUTE}--5", [5]),
    (f"{TEST_ROUTE}/0", [0]),
    (f"{TEST_ROUTE}/5", [5]),
    (f"{TEST_ROUTE}/0:10", ALL_SEGS[0:10]),
    (f"{TEST_ROUTE}/0:0", []),
    (f"{TEST_ROUTE}/4:6", ALL_SEGS[4:6]),
    (f"{TEST_ROUTE}/0:-1", ALL_SEGS[0:-1]),
    (f"{TEST_ROUTE}/:5", ALL_SEGS[:5]),
    (f"{TEST_ROUTE}/2:", ALL_SEGS[2:]),
    (f"{TEST_ROUTE}/2:-1", ALL_SEGS[2:-1]),
    (f"{TEST_ROUTE}/-1", [ALL_SEGS[-1]]),
    (f"{TEST_ROUTE}/-2", [ALL_SEGS[-2]]),
    (f"{TEST_ROUTE}/-2:-1", ALL_SEGS[-2:-1]),
    (f"{TEST_ROUTE}/-4:-2", ALL_SEGS[-4:-2]),
    (f"{TEST_ROUTE}/:10:2", ALL_SEGS[:10:2]),
    (f"{TEST_ROUTE}/5::2", ALL_SEGS[5::2]),
  ])
  def test_parse_slice(self, segment_range, expected):
    sr = SegmentRange(segment_range)
    segs = parse_slice(sr)
    self.assertListEqual(list(segs), expected)

  @parameterized.expand([
    (f"{TEST_ROUTE}//",),
    (f"{TEST_ROUTE}---",),
    (f"{TEST_ROUTE}/-4:--2",),
    (f"{TEST_ROUTE}/-a",),
    (f"{TEST_ROUTE}/0:1:2:3",),
    (f"{TEST_ROUTE}/:::3",),
  ])
  def test_bad_ranges(self, segment_range):
    with self.assertRaises(AssertionError):
      sr = SegmentRange(segment_range)
      parse_slice(sr)

  @parameterized.expand([
    (ReadMode.QLOG, 11643),
    (ReadMode.RLOG, 70577),
  ])
  def test_modes(self, mode, expected):
    lr = SegmentRangeReader(TEST_ROUTE+"/0", mode)

    self.assertEqual(len(list(lr)), expected)


if __name__ == "__main__":
  unittest.main()
