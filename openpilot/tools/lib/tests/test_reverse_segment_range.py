import unittest
from unittest.mock import patch

from openpilot.tools.lib.route import SegmentRange


class TestReverseSegmentRange(unittest.TestCase):
  def test_explicit_start(self):
    route = "a2a0ccea32023010/2023-07-27--13-01-19"
    with patch("openpilot.tools.lib.route.get_max_seg_number_cached") as maximum:
      self.assertEqual(SegmentRange(f"{route}/5:1:-2").seg_idxs, [5, 3])
      maximum.assert_not_called()

  def test_start_at_route_end(self):
    route = "a2a0ccea32023010/2023-07-27--13-01-19"
    with patch("openpilot.tools.lib.route.get_max_seg_number_cached", return_value=7):
      self.assertEqual(SegmentRange(f"{route}/:2:-2").seg_idxs, [7, 5, 3])
