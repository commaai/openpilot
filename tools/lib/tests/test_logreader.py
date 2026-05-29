import capnp
import contextlib
import io
import shutil
import tempfile
import os
import pytest
import requests

from openpilot.common.parameterized import parameterized

from cereal import log as capnp_log
from openpilot.tools.lib.logreader import LogsUnavailable, LogIterable, LogReader, parse_indirect, ReadMode, auto_camera_source
from openpilot.tools.lib.file_sources import comma_api_source, InternalUnavailableException
from openpilot.tools.lib.route import SegmentRange, FileName
from openpilot.tools.lib.url_file import URLFileException

NUM_SEGS = 17  # number of segments in the test route
ALL_SEGS = list(range(NUM_SEGS))
TEST_ROUTE = "344c5c15b34f2d8a/2024-01-03--09-37-12"
QLOG_FILE = "https://commadataci.blob.core.windows.net/openpilotci/0375fdf7b1ce594d/2019-06-13--08-32-25/3/qlog.bz2"


def noop(segment: LogIterable):
  return segment


@contextlib.contextmanager
def setup_source_scenario(mocker, is_internal=False):
  internal_source_mock = mocker.patch("openpilot.tools.lib.logreader.internal_source")
  internal_source_mock.__name__ = internal_source_mock._mock_name

  openpilotci_source_mock = mocker.patch("openpilot.tools.lib.logreader.openpilotci_source")
  openpilotci_source_mock.__name__ = openpilotci_source_mock._mock_name

  comma_api_source_mock = mocker.patch("openpilot.tools.lib.logreader.comma_api_source")
  comma_api_source_mock.__name__ = comma_api_source_mock._mock_name

  if is_internal:
    internal_source_mock.return_value = {3: QLOG_FILE}
  else:
    internal_source_mock.side_effect = InternalUnavailableException

  openpilotci_source_mock.return_value = {}
  comma_api_source_mock.return_value = {3: QLOG_FILE}

  yield


class TestLogReader:
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
    (f"https://useradmin.comma.ai/?onebox={TEST_ROUTE}", ALL_SEGS),
    (f"https://useradmin.comma.ai/?onebox={TEST_ROUTE.replace('/', '|')}", ALL_SEGS),
    (f"https://useradmin.comma.ai/?onebox={TEST_ROUTE.replace('/', '%7C')}", ALL_SEGS),
  ])
  @pytest.mark.skip("this got flaky. internet tests are stupid.")
  def test_indirect_parsing(self, identifier, expected):
    parsed = parse_indirect(identifier)
    sr = SegmentRange(parsed)
    assert list(sr.seg_idxs) == expected, identifier

  @parameterized.expand([
    (f"{TEST_ROUTE}", f"{TEST_ROUTE}"),
    (f"{TEST_ROUTE.replace('/', '|')}", f"{TEST_ROUTE}"),
    (f"{TEST_ROUTE}--5", f"{TEST_ROUTE}/5"),
    (f"{TEST_ROUTE}/0/q", f"{TEST_ROUTE}/0/q"),
    (f"{TEST_ROUTE}/5:6/r", f"{TEST_ROUTE}/5:6/r"),
    (f"{TEST_ROUTE}/5", f"{TEST_ROUTE}/5"),
  ])
  def test_canonical_name(self, identifier, expected):
    sr = SegmentRange(identifier)
    assert str(sr) == expected

  @pytest.mark.parametrize("cache_enabled", [True, False])
  def test_direct_parsing(self, mocker, cache_enabled):
    file_exists_mock = mocker.patch("openpilot.tools.lib.filereader.file_exists")
    if cache_enabled:
      os.environ.pop("DISABLE_FILEREADER_CACHE", None)
    else:
      os.environ["DISABLE_FILEREADER_CACHE"] = "1"
    qlog = tempfile.NamedTemporaryFile(mode='wb', delete=False)

    with requests.get(QLOG_FILE, stream=True) as r:
      with qlog as f:
        shutil.copyfileobj(r.raw, f)

    for f in [QLOG_FILE, qlog.name]:
      l = len(list(LogReader(f)))
      assert l > 100

    with pytest.raises(URLFileException) if not cache_enabled else pytest.raises(AssertionError):
      l = len(list(LogReader(QLOG_FILE.replace("/3/", "/200/"))))

    # file_exists should not be called for direct files
    assert file_exists_mock.call_count == 0

  @parameterized.expand([
    (f"{TEST_ROUTE}///",),
    (f"{TEST_ROUTE}---",),
    (f"{TEST_ROUTE}/-4:--2",),
    (f"{TEST_ROUTE}/-a",),
    (f"{TEST_ROUTE}/j",),
    (f"{TEST_ROUTE}/0:1:2:3",),
    (f"{TEST_ROUTE}/:::3",),
    (f"{TEST_ROUTE}3",),
    (f"{TEST_ROUTE}-3",),
    (f"{TEST_ROUTE}--3a",),
  ])
  def test_bad_ranges(self, segment_range):
    with pytest.raises(AssertionError):
      _ = SegmentRange(segment_range).seg_idxs

  @pytest.mark.parametrize("segment_range, api_call", [
    (f"{TEST_ROUTE}/0", False),
    (f"{TEST_ROUTE}/:2", False),
    (f"{TEST_ROUTE}/0:", True),
    (f"{TEST_ROUTE}/-1", True),
    (f"{TEST_ROUTE}", True),
  ])
  def test_slicing_api_call(self, mocker, segment_range, api_call):
    max_seg_mock = mocker.patch("openpilot.tools.lib.route.get_max_seg_number_cached")
    max_seg_mock.return_value = NUM_SEGS
    _ = SegmentRange(segment_range).seg_idxs
    assert api_call == max_seg_mock.called

  @pytest.mark.slow
  def test_modes(self):
    qlog_len = len(list(LogReader(f"{TEST_ROUTE}/0", ReadMode.QLOG)))
    rlog_len = len(list(LogReader(f"{TEST_ROUTE}/0", ReadMode.RLOG)))

    assert qlog_len * 6 < rlog_len

  @pytest.mark.slow
  def test_modes_from_name(self):
    qlog_len = len(list(LogReader(f"{TEST_ROUTE}/0/q")))
    rlog_len = len(list(LogReader(f"{TEST_ROUTE}/0/r")))

    assert qlog_len * 6 < rlog_len

  @pytest.mark.slow
  def test_list(self):
    qlog_len = len(list(LogReader(f"{TEST_ROUTE}/0/q")))
    qlog_len_2 = len(list(LogReader([f"{TEST_ROUTE}/0/q", f"{TEST_ROUTE}/0/q"])))

    assert qlog_len * 2 == qlog_len_2

  @pytest.mark.slow
  def test_multiple_iterations(self, mocker):
    init_mock = mocker.patch("openpilot.tools.lib.logreader._LogFileReader")
    lr = LogReader(f"{TEST_ROUTE}/0/q")
    qlog_len1 = len(list(lr))
    qlog_len2 = len(list(lr))

    # ensure we don't create multiple instances of _LogFileReader, which means downloading the files twice
    assert init_mock.call_count == 1

    assert qlog_len1 == qlog_len2

  @pytest.mark.slow
  def test_helpers(self):
    lr = LogReader(f"{TEST_ROUTE}/0/q")
    assert lr.first("carParams").carFingerprint == "SUBARU OUTBACK 6TH GEN"
    assert 0 < len(list(lr.filter("carParams"))) < len(list(lr))

  @parameterized.expand([(True,), (False,)])
  @pytest.mark.slow
  def test_run_across_segments(self, cache_enabled):
    if cache_enabled:
      os.environ.pop("DISABLE_FILEREADER_CACHE", None)
    else:
      os.environ["DISABLE_FILEREADER_CACHE"] = "1"
    lr = LogReader(f"{TEST_ROUTE}/0:4")
    assert len(lr.run_across_segments(4, noop)) == len(list(lr))

  @pytest.mark.slow
  def test_auto_mode(self, subtests, mocker):
    lr = LogReader(f"{TEST_ROUTE}/0/q")
    qlog_len = len(list(lr))
    log_paths_mock = mocker.patch("openpilot.tools.lib.route.Route.log_paths")
    log_paths_mock.return_value = [None] * NUM_SEGS
    # Should fall back to qlogs since rlogs are not available

    with subtests.test("interactive_yes"):
      mocker.patch("sys.stdin", new=io.StringIO("y\n"))
      lr = LogReader(f"{TEST_ROUTE}/0", default_mode=ReadMode.AUTO_INTERACTIVE, sources=[comma_api_source])
      log_len = len(list(lr))
      assert qlog_len == log_len

    with subtests.test("interactive_no"):
      mocker.patch("sys.stdin", new=io.StringIO("n\n"))
      with pytest.raises(LogsUnavailable):
        lr = LogReader(f"{TEST_ROUTE}/0", default_mode=ReadMode.AUTO_INTERACTIVE, sources=[comma_api_source])

    with subtests.test("non_interactive"):
      lr = LogReader(f"{TEST_ROUTE}/0", default_mode=ReadMode.AUTO, sources=[comma_api_source])
      log_len = len(list(lr))
      assert qlog_len == log_len

  @pytest.mark.parametrize("is_internal", [True, False])
  def test_auto_source_scenarios(self, mocker, is_internal):
    lr = LogReader(QLOG_FILE)
    qlog_len = len(list(lr))

    with setup_source_scenario(mocker, is_internal=is_internal):
      lr = LogReader(f"{TEST_ROUTE}/3/q")
      log_len = len(list(lr))
      assert qlog_len == log_len

  @pytest.mark.slow
  def test_sort_by_time(self):
    msgs = list(LogReader(f"{TEST_ROUTE}/0/q"))
    assert msgs != sorted(msgs, key=lambda m: m.logMonoTime)

    msgs = list(LogReader(f"{TEST_ROUTE}/0/q", sort_by_time=True))
    assert msgs == sorted(msgs, key=lambda m: m.logMonoTime)

  def test_only_union_types(self):
    with tempfile.NamedTemporaryFile() as qlog:
      # write valid Event messages
      num_msgs = 100
      with open(qlog.name, "wb") as f:
        f.write(b"".join(capnp_log.Event.new_message().to_bytes() for _ in range(num_msgs)))

      msgs = list(LogReader(qlog.name))
      assert len(msgs) == num_msgs
      [m.which() for m in msgs]

      # append non-union Event message
      event_msg = capnp_log.Event.new_message()
      non_union_bytes = bytearray(event_msg.to_bytes())
      non_union_bytes[event_msg.total_size.word_count * 8] = 0xff  # set discriminant value out of range using Event word offset
      with open(qlog.name, "ab") as f:
        f.write(non_union_bytes)

      # ensure new message is added, but is not a union type
      msgs = list(LogReader(qlog.name))
      assert len(msgs) == num_msgs + 1
      with pytest.raises(capnp.KjException):
        [m.which() for m in msgs]

      # should not be added when only_union_types=True
      msgs = list(LogReader(qlog.name, only_union_types=True))
      assert len(msgs) == num_msgs
      [m.which() for m in msgs]


TEST_ROUTE_CAM = "344c5c15b34f2d8a/2024-01-03--09-37-12"
CAMERA_FILE = "fcamera.hevc"


class TestCameraSource:
  def test_auto_camera_source_no_source(self):
    with pytest.raises(LogsUnavailable):
      auto_camera_source(f"{TEST_ROUTE_CAM}/0", [], FileName.FCAMERA)

  def test_auto_camera_source_no_source_default_camera(self):
    with pytest.raises(LogsUnavailable):
      auto_camera_source(f"{TEST_ROUTE_CAM}/0", [])

  def test_auto_camera_source_found(self, mocker):
    mock_source = mocker.Mock()
    mock_source.__name__ = "mock_source"
    mock_source.return_value = {0: "http://fake/fcamera.hevc"}

    result = auto_camera_source(f"{TEST_ROUTE_CAM}/0", [mock_source], FileName.FCAMERA)
    assert len(result) == 1
    assert result[0] == "http://fake/fcamera.hevc"

  def test_auto_camera_source_multiple_segments(self, mocker):
    mock_source = mocker.Mock()
    mock_source.__name__ = "mock_source"
    mock_source.return_value = {0: "http://fake/0/fcamera.hevc", 1: "http://fake/1/fcamera.hevc"}

    result = auto_camera_source(f"{TEST_ROUTE_CAM}/0:2", [mock_source], FileName.FCAMERA)
    assert len(result) == 2

  def test_auto_camera_source_source_fallback(self, mocker):
    source_a = mocker.Mock()
    source_a.__name__ = "source_a"
    source_a.return_value = {}

    source_b = mocker.Mock()
    source_b.__name__ = "source_b"
    source_b.return_value = {0: "http://fake/fcamera.hevc"}

    result = auto_camera_source(f"{TEST_ROUTE_CAM}/0", [source_a, source_b], FileName.FCAMERA)
    assert len(result) == 1
    assert source_a.called
    assert source_b.called

  def test_auto_camera_source_partial_fallback(self, mocker):
    source_a = mocker.Mock()
    source_a.__name__ = "source_a"
    source_a.return_value = {0: "http://fake/0/fcamera.hevc"}

    source_b = mocker.Mock()
    source_b.__name__ = "source_b"
    source_b.return_value = {1: "http://fake/1/fcamera.hevc"}

    result = auto_camera_source(f"{TEST_ROUTE_CAM}/0:2", [source_a, source_b], FileName.FCAMERA)
    assert len(result) == 2

  def test_auto_camera_source_error_then_success(self, mocker):
    source_a = mocker.Mock()
    source_a.__name__ = "source_a"
    source_a.side_effect = Exception("network error")

    source_b = mocker.Mock()
    source_b.__name__ = "source_b"
    source_b.return_value = {0: "http://fake/fcamera.hevc"}

    result = auto_camera_source(f"{TEST_ROUTE_CAM}/0", [source_a, source_b], FileName.FCAMERA)
    assert len(result) == 1

  def test_auto_camera_source_all_fail(self, mocker):
    source_a = mocker.Mock()
    source_a.__name__ = "source_a"
    source_a.side_effect = Exception("network error")

    source_b = mocker.Mock()
    source_b.__name__ = "source_b"
    source_b.return_value = {}

    with pytest.raises(LogsUnavailable):
      auto_camera_source(f"{TEST_ROUTE_CAM}/0:2", [source_a, source_b], FileName.FCAMERA)

  def test_auto_camera_source_route_with_slash(self, mocker):
    mocker.patch("openpilot.tools.lib.route.get_max_seg_number_cached", return_value=16)
    mock_source = mocker.Mock()
    mock_source.__name__ = "mock_source"
    mock_source.side_effect = lambda sr, seg_idxs, fns: {seg: f"http://fake/{seg}/fcamera.hevc" for seg in seg_idxs}

    result = auto_camera_source(f"{TEST_ROUTE_CAM.replace('/', '|')}", [mock_source], FileName.FCAMERA)
    assert len(result) == 17

  def test_auto_camera_source_route_with_segment(self, mocker):
    mock_source = mocker.Mock()
    mock_source.__name__ = "mock_source"
    mock_source.side_effect = lambda sr, seg_idxs, fns: {seg: f"http://fake/{seg}/fcamera.hevc" for seg in seg_idxs}

    result = auto_camera_source(f"{TEST_ROUTE_CAM}--5", [mock_source], FileName.FCAMERA)
    assert len(result) == 1

  def test_comma_api_source_extended(self, mocker):
    """Verify comma_api_source can serve camera paths"""
    mock_route = mocker.patch("openpilot.tools.lib.file_sources.Route")
    mock_route_instance = mock_route.return_value
    mock_route_instance.log_paths.return_value = ["fake_rlog"]
    mock_route_instance.qlog_paths.return_value = ["fake_qlog"]
    mock_route_instance.camera_paths.return_value = ["fake_fcamera"]
    mock_route_instance.dcamera_paths.return_value = ["fake_dcamera"]
    mock_route_instance.ecamera_paths.return_value = ["fake_ecamera"]
    mock_route_instance.qcamera_paths.return_value = ["fake_qcamera"]

    sr = SegmentRange(f"{TEST_ROUTE_CAM}/0")
    for fns, expected in [
      (FileName.RLOG, "fake_rlog"),
      (FileName.QLOG, "fake_qlog"),
      (FileName.FCAMERA, "fake_fcamera"),
      (FileName.DCAMERA, "fake_dcamera"),
      (FileName.ECAMERA, "fake_ecamera"),
      (FileName.QCAMERA, "fake_qcamera"),
    ]:
      result = comma_api_source(sr, [0], fns)
      assert result[0] == expected, f"Failed for {fns}"

  def test_comma_api_source_unknown_type(self, mocker):
    mocker.patch("openpilot.tools.lib.file_sources.Route")
    sr = SegmentRange(f"{TEST_ROUTE_CAM}/0")
    with pytest.raises(ValueError, match="Unknown file type"):
      comma_api_source(sr, [0], ("unknown.ext",))
