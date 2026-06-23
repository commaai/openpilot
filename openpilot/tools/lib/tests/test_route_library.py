from collections import namedtuple

from openpilot.tools.lib.route import SegmentName

class TestRouteLibrary:
  def test_segment_name_formats(self):
    Case = namedtuple('Case', ['input', 'expected_route', 'expected_segment_num', 'expected_data_dir'])

    cases = [ Case("a2a0ccea32023010|2023-07-27--13-01-19", "a2a0ccea32023010|2023-07-27--13-01-19", -1, None),
              Case("a2a0ccea32023010/2023-07-27--13-01-19--1", "a2a0ccea32023010|2023-07-27--13-01-19", 1, None),
              Case("a2a0ccea32023010|2023-07-27--13-01-19/2", "a2a0ccea32023010|2023-07-27--13-01-19", 2, None),
              Case("a2a0ccea32023010/2023-07-27--13-01-19/3", "a2a0ccea32023010|2023-07-27--13-01-19", 3, None),
              Case("/data/media/0/realdata/a2a0ccea32023010|2023-07-27--13-01-19", "a2a0ccea32023010|2023-07-27--13-01-19", -1, "/data/media/0/realdata"),
              Case("/data/media/0/realdata/a2a0ccea32023010|2023-07-27--13-01-19--1", "a2a0ccea32023010|2023-07-27--13-01-19", 1, "/data/media/0/realdata"),
              Case("/data/media/0/realdata/a2a0ccea32023010|2023-07-27--13-01-19/2", "a2a0ccea32023010|2023-07-27--13-01-19", 2, "/data/media/0/realdata") ]

    def _validate(case):
      route_or_segment_name = case.input

      s = SegmentName(route_or_segment_name, allow_route_name=True)

      assert str(s.route_name) == case.expected_route
      assert s.segment_num == case.expected_segment_num
      assert s.data_dir == case.expected_data_dir

    for case in cases:
      _validate(case)
