import numpy as np
import unittest
from parameterized import parameterized

from openpilot.tools.lib.route import Route, SegmentRange
from openpilot.tools.lib.srreader import parse_start_end

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
    (f"{TEST_ROUTE}/0/10", list(np.arange(0, 10))),
    (f"{TEST_ROUTE}/0/0", []),
    (f"{TEST_ROUTE}/4/6", [4,5]),
    (f"{TEST_ROUTE}/0/-1", ALL_SEGS[:-1]),
    (f"{TEST_ROUTE}/2/-1", ALL_SEGS[2:-1]),
    (f"{TEST_ROUTE}/-1", [NUM_SEGS-1]),
    (f"{TEST_ROUTE}/-2/-1", [NUM_SEGS-2]),
    (f"{TEST_ROUTE}/-4/-2", [NUM_SEGS-4, NUM_SEGS-3]),
  ])
  def test_parse_start_end(self, segment_range, expected):
    sr = SegmentRange(segment_range)
    route = Route(sr.route_name)
    segs = parse_start_end(sr, route)
    self.assertListEqual(list(segs), expected)


if __name__ == "__main__":
  unittest.main()
