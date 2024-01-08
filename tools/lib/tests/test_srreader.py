import unittest
from parameterized import parameterized
from openpilot.tools.lib.route import Route, SegmentRange

from openpilot.tools.lib.srreader import parse_start_end

NUM_SEGS = 17

class TestSegmentRangeReader(unittest.TestCase):
  @parameterized.expand([
    ("344c5c15b34f2d8a|2024-01-03--09-37-12", (0, NUM_SEGS-1)),
    ("344c5c15b34f2d8a/2024-01-03--09-37-12", (0, NUM_SEGS-1)),
    ("344c5c15b34f2d8a|2024-01-03--09-37-12--0", (0, 0)),
    ("344c5c15b34f2d8a|2024-01-03--09-37-12/0", (0, 0)),
    ("344c5c15b34f2d8a|2024-01-03--09-37-12/0/10", (0, 10)),
    ("344c5c15b34f2d8a|2024-01-03--09-37-12/0/-1", (0, NUM_SEGS-1)),
    ("344c5c15b34f2d8a|2024-01-03--09-37-12/-1", (NUM_SEGS-1, NUM_SEGS-1)),
    ("344c5c15b34f2d8a|2024-01-03--09-37-12/-2/-1", (NUM_SEGS-2, NUM_SEGS-1)),
  ])
  def test_parse_start_end(self, segment_range, expected):
    sr = SegmentRange(segment_range)
    route = Route(sr.route_name)
    actual = parse_start_end(sr, route)
    self.assertEqual(actual, expected)

if __name__ == "__main__":
  unittest.main()
