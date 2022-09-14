#!/usr/bin/env python
import unittest
from collections import namedtuple

from tools.lib.route import SegmentName

class TestRouteLibrary(unittest.TestCase):
  def test_segment_name_formats(self):
    Case = namedtuple('Case', ['input', 'expected_route', 'expected_segment_num', 'expected_data_dir'])

    cases = [ Case("4cf7a6ad03080c90|2021-09-29--13-46-36", "4cf7a6ad03080c90|2021-09-29--13-46-36", -1, None),
              Case("4cf7a6ad03080c90/2021-09-29--13-46-36--1", "4cf7a6ad03080c90|2021-09-29--13-46-36", 1, None),
              Case("4cf7a6ad03080c90|2021-09-29--13-46-36/2", "4cf7a6ad03080c90|2021-09-29--13-46-36", 2, None),
              Case("4cf7a6ad03080c90/2021-09-29--13-46-36/3", "4cf7a6ad03080c90|2021-09-29--13-46-36", 3, None),
              Case("/data/media/0/realdata/4cf7a6ad03080c90|2021-09-29--13-46-36", "4cf7a6ad03080c90|2021-09-29--13-46-36", -1, "/data/media/0/realdata"),
              Case("/data/media/0/realdata/4cf7a6ad03080c90|2021-09-29--13-46-36--1", "4cf7a6ad03080c90|2021-09-29--13-46-36", 1, "/data/media/0/realdata"),
              Case("/data/media/0/realdata/4cf7a6ad03080c90|2021-09-29--13-46-36/2", "4cf7a6ad03080c90|2021-09-29--13-46-36", 2, "/data/media/0/realdata") ]

    def _validate(case):
      route_or_segment_name = case.input

      s = SegmentName(route_or_segment_name, allow_route_name=True)

      self.assertEqual(str(s.route_name), case.expected_route)
      self.assertEqual(s.segment_num, case.expected_segment_num)
      self.assertEqual(s.data_dir, case.expected_data_dir)

    for case in cases:
      _validate(case)

if __name__ == "__main__":
  unittest.main()
