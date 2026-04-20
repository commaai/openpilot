from collections import namedtuple
import importlib
import sys
from types import ModuleType


def _import_route_module():
  if "openpilot.tools.lib.auth_config" not in sys.modules:
    auth_config = ModuleType("openpilot.tools.lib.auth_config")
    auth_config.get_token = lambda: "test-token"
    sys.modules["openpilot.tools.lib.auth_config"] = auth_config
  return importlib.import_module("openpilot.tools.lib.route")

class TestRouteLibrary:
  def test_segment_name_formats(self):
    route_lib = _import_route_module()

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

      s = route_lib.SegmentName(route_or_segment_name, allow_route_name=True)

      assert str(s.route_name) == case.expected_route
      assert s.segment_num == case.expected_segment_num
      assert s.data_dir == case.expected_data_dir

    for case in cases:
      _validate(case)

  def test_local_segment_discovery_across_layouts(self, tmp_path):
    route_lib = _import_route_module()

    route_name = "a2a0ccea32023010|2023-07-27--13-01-19"

    explorer_rlog = tmp_path / f"{route_name}--0--rlog.zst"
    explorer_rlog.write_text("")

    op_segment_dir = tmp_path / f"{route_name}--1"
    op_segment_dir.mkdir()
    op_qlog = op_segment_dir / "qlog.bz2"
    op_qlog.write_text("")
    op_fcamera = op_segment_dir / "fcamera.hevc"
    op_fcamera.write_text("")

    canonical_segment_dir = tmp_path / route_name / "2"
    canonical_segment_dir.mkdir(parents=True)
    canonical_dcamera = canonical_segment_dir / "dcamera.hevc"
    canonical_dcamera.write_text("")
    canonical_ecamera = canonical_segment_dir / "ecamera.hevc"
    canonical_ecamera.write_text("")

    route = route_lib.Route(route_name, data_dir=str(tmp_path))

    assert route.log_paths() == [str(explorer_rlog), None, None]
    assert route.qlog_paths() == [None, str(op_qlog), None]
    assert route.camera_paths() == [None, str(op_fcamera), None]
    assert route.dcamera_paths() == [None, None, str(canonical_dcamera)]
    assert route.ecamera_paths() == [None, None, str(canonical_ecamera)]
    assert route.qcamera_paths() == [None, None, None]

  def test_remote_file_mapping_preserves_precedence(self, monkeypatch):
    route_lib = _import_route_module()

    route_name = "a2a0ccea32023010|2023-07-27--13-01-19"
    seg0_rlog_bz2 = "https://example.com/a2a0ccea32023010/2023-07-27--13-01-19/0/rlog.bz2"
    seg0_rlog_zst = "https://example.com/a2a0ccea32023010/2023-07-27--13-01-19/0/rlog.zst"
    seg1_qlog = "https://example.com/a2a0ccea32023010/2023-07-27--13-01-19/1/qlog.bz2"
    seg2_qcamera = "https://example.com/a2a0ccea32023010/2023-07-27--13-01-19/2/qcamera.ts"

    route_files = {
      "logs": [seg0_rlog_bz2, seg0_rlog_zst, seg1_qlog],
      "cameras": [seg2_qcamera],
    }

    class FakeApi:
      def __init__(self, token):
        assert token == "fake-token"

      def get(self, endpoint):
        assert endpoint == f"v1/route/{route_name}/files"
        return route_files

    monkeypatch.setattr(route_lib, "get_token", lambda: "fake-token")
    monkeypatch.setattr(route_lib, "CommaApi", FakeApi)

    route = route_lib.Route(route_name)

    assert route.log_paths() == [seg0_rlog_zst, None, None]
    assert route.qlog_paths() == [None, seg1_qlog, None]
    assert route.qcamera_paths() == [None, None, seg2_qcamera]

  def test_sparse_segments_keep_none_holes(self, monkeypatch):
    route_lib = _import_route_module()

    route_name = "a2a0ccea32023010|2023-07-27--13-01-19"
    seg0_rlog = "https://example.com/a2a0ccea32023010/2023-07-27--13-01-19/0/rlog.zst"
    seg2_rlog = "https://example.com/a2a0ccea32023010/2023-07-27--13-01-19/2/rlog.zst"

    class FakeApi:
      def __init__(self, token):
        assert token == "fake-token"

      def get(self, endpoint):
        assert endpoint == f"v1/route/{route_name}/files"
        return {"logs": [seg0_rlog, seg2_rlog]}

    monkeypatch.setattr(route_lib, "get_token", lambda: "fake-token")
    monkeypatch.setattr(route_lib, "CommaApi", FakeApi)

    route = route_lib.Route(route_name)
    assert route.log_paths() == [seg0_rlog, None, seg2_rlog]
