from collections import namedtuple

from openpilot.common.test import OpenpilotTestCase
from openpilot.tools.lib.route import RouteName, SegmentName, SegmentRange

class TestRouteLibrary(OpenpilotTestCase):
  def test_segment_name_formats(self):
    Case = namedtuple('Case', ['input', 'expected_route', 'expected_segment_num', 'expected_data_dir'])

    cases = [ Case("a2a0ccea32023010|2023-07-27--13-01-19", "a2a0ccea32023010/2023-07-27--13-01-19", -1, None),
              Case("a2a0ccea32023010/2023-07-27--13-01-19--1", "a2a0ccea32023010/2023-07-27--13-01-19", 1, None),
              Case("a2a0ccea32023010|2023-07-27--13-01-19/2", "a2a0ccea32023010/2023-07-27--13-01-19", 2, None),
              Case("a2a0ccea32023010/2023-07-27--13-01-19/3", "a2a0ccea32023010/2023-07-27--13-01-19", 3, None),
              Case("/data/media/0/realdata/a2a0ccea32023010|2023-07-27--13-01-19", "a2a0ccea32023010/2023-07-27--13-01-19", -1, "/data/media/0/realdata"),
              Case("/data/media/0/realdata/a2a0ccea32023010|2023-07-27--13-01-19--1", "a2a0ccea32023010/2023-07-27--13-01-19", 1, "/data/media/0/realdata"),
              Case("/data/media/0/realdata/a2a0ccea32023010|2023-07-27--13-01-19/2", "a2a0ccea32023010/2023-07-27--13-01-19", 2, "/data/media/0/realdata") ]

    def _validate(case):
      route_or_segment_name = case.input

      s = SegmentName(route_or_segment_name, allow_route_name=True)

      assert str(s.route_name) == case.expected_route
      assert s.segment_num == case.expected_segment_num
      assert s.data_dir == case.expected_data_dir

    for case in cases:
      _validate(case)

  def test_slash_output(self):
    for log_id in ("2023-07-27--13-01-19", "000000cc--8e8d8ec716"):
      route = f"a2a0ccea32023010/{log_id}"
      for separator in ("/", "|"):
        legacy = route.replace("/", separator)
        assert str(RouteName(legacy)) == route
        assert SegmentRange(legacy + "/1:3").route_name == route
        assert str(SegmentRange(legacy + "/1:3")) == route + "/1:3"
        assert str(SegmentName(legacy, allow_route_name=True)) == route + "---1"
        for segment_separator in ("/", "--"):
          name = legacy + segment_separator + "2"
          segment = SegmentName(name)
          assert str(segment) == route + "--2"
          assert segment.data_name == route + "/2"
          local = SegmentName("/data/" + name)
          assert local.data_dir == "/data"
          assert str(local) == str(segment)
