from pathlib import Path
import subprocess

import pytest

from openpilot.cereal import log, messaging
import openpilot.tools.cm5.validate_route as validator
from openpilot.tools.cm5.validate_route import (
  PANDA_STATE_MAX_GAP_NS,
  RouteValidationError,
  _probe_video,
  _require_frame_count,
  _require_panda_coverage,
  discover_segments,
  validate_route,
)
from openpilot.tools.lib.logreader import LogReader, save_log


def event(message_type: str, mono_time: int, *, capture_timestamps: bool = True):
  if message_type == "can":
    message = messaging.new_message("can", 1)
    message.can[0].address = 0x123
    message.can[0].dat = b"\x01\x02"
    message.can[0].src = 0
  elif message_type == "roadEncodeIdx":
    message = messaging.new_message("roadEncodeIdx")
    message.roadEncodeIdx.frameId = 1
    message.roadEncodeIdx.encodeId = 1
    message.roadEncodeIdx.segmentId = 1
    message.roadEncodeIdx.type = log.EncodeIndex.Type.fullH264
    if capture_timestamps:
      message.roadEncodeIdx.timestampSof = mono_time - 1
      message.roadEncodeIdx.timestampEof = mono_time
  elif message_type == "qRoadEncodeIdx":
    message = messaging.new_message("qRoadEncodeIdx")
    message.qRoadEncodeIdx.frameId = 1
    message.qRoadEncodeIdx.encodeId = 1
    message.qRoadEncodeIdx.segmentId = 1
    message.qRoadEncodeIdx.type = log.EncodeIndex.Type.qcameraH264
    if capture_timestamps:
      message.qRoadEncodeIdx.timestampSof = mono_time - 1
      message.qRoadEncodeIdx.timestampEof = mono_time
  elif message_type == "roadCameraState":
    message = messaging.new_message("roadCameraState")
    if capture_timestamps:
      message.roadCameraState.timestampSof = mono_time - 1
      message.roadCameraState.timestampEof = mono_time
  elif message_type == "pandaStates":
    message = messaging.new_message("pandaStates", 1)
    message.pandaStates[0].pandaType = "redPanda"
    message.pandaStates[0].safetyModel = "silent"
    message.pandaStates[0].controlsAllowed = False
  else:
    message = messaging.new_message(message_type)
  message.logMonoTime = mono_time
  return message


def make_segment(root: Path, prefix: str, number: int, *, can=True, camera=True, passive=True,
                 capture_timestamps=True, ending_sentinel=True) -> Path:
  segment = root / f"{prefix}--{number}"
  segment.mkdir()
  messages = [event("initData", 1)]
  messages[0].initData.deviceType = "pc"
  messages[0].initData.passive = passive
  if can:
    messages += [event("can", 100), event("pandaStates", 150), event("can", 300)]
  if camera:
    messages += [
      event("roadCameraState", 200, capture_timestamps=capture_timestamps),
      event("roadEncodeIdx", 250, capture_timestamps=capture_timestamps),
      event("qRoadEncodeIdx", 250, capture_timestamps=capture_timestamps),
    ]
  if ending_sentinel:
    messages.append(event("sentinel", 400))
    messages[-1].sentinel.type = log.Sentinel.SentinelType.endOfSegment
  readers = [message.as_reader() for message in messages]
  save_log(str(segment / "rlog.zst"), readers)
  save_log(str(segment / "qlog.zst"), readers)
  (segment / "fcamera.hevc").write_bytes(b"video")
  (segment / "qcamera.ts").write_bytes(b"qvideo")
  return segment


def test_validate_complete_route(tmp_path: Path):
  prefix = "00000000--0123456789"
  first = make_segment(tmp_path, prefix, 0)
  make_segment(tmp_path, prefix, 1)

  report = validate_route(first)

  assert report.prefix == tmp_path / prefix
  assert len(report.segments) == 2
  assert report.can_frames == 4
  assert report.encoded_frames == 2
  assert not report.trimmed


def test_validate_contiguous_retained_suffix(tmp_path: Path):
  prefix = "00000000--0123456789"
  first = make_segment(tmp_path, prefix, 8)
  make_segment(tmp_path, prefix, 9)

  report = validate_route(first)

  assert report.trimmed
  assert [segment.number for segment in report.segments] == [8, 9]


def test_rejects_missing_can(tmp_path: Path):
  segment = make_segment(tmp_path, "00000000--0123456789", 0, can=False)

  with pytest.raises(RouteValidationError, match="no CAN frames"):
    validate_route(segment)


def test_rejects_missing_required_can_bus(tmp_path: Path):
  segment = make_segment(tmp_path, "00000000--0123456789", 0)

  with pytest.raises(RouteValidationError, match="required CAN bus 1 has no frames"):
    validate_route(segment, required_can_buses=frozenset({1}))


def test_rejects_non_passive_route(tmp_path: Path):
  segment = make_segment(tmp_path, "00000000--0123456789", 0, passive=False)

  with pytest.raises(RouteValidationError, match="not marked passive"):
    validate_route(segment)


def test_rejects_non_silent_panda_state(tmp_path: Path):
  segment = make_segment(tmp_path, "00000000--0123456789", 0)
  messages = list(LogReader(str(segment / "rlog.zst")))
  mutable = [message.as_builder() for message in messages]
  panda_state = next(message.pandaStates[0] for message in mutable if message.which() == "pandaStates")
  panda_state.safetyModel = "noOutput"
  save_log(str(segment / "rlog.zst"), [message.as_reader() for message in mutable])

  with pytest.raises(RouteValidationError, match="SILENT receive-only"):
    validate_route(segment)


def test_rejects_panda_transmission_counter(tmp_path: Path):
  segment = make_segment(tmp_path, "00000000--0123456789", 0)
  messages = list(LogReader(str(segment / "rlog.zst")))
  mutable = [message.as_builder() for message in messages]
  panda_state = next(message.pandaStates[0] for message in mutable if message.which() == "pandaStates")
  panda_state.canState1.totalFwdCnt = 1
  save_log(str(segment / "rlog.zst"), [message.as_reader() for message in mutable])

  with pytest.raises(RouteValidationError, match="transmitted or forwarded"):
    validate_route(segment)


def test_rejects_panda_receive_loss_counter(tmp_path: Path):
  segment = make_segment(tmp_path, "00000000--0123456789", 0)
  messages = list(LogReader(str(segment / "rlog.zst")))
  mutable = [message.as_builder() for message in messages]
  panda_state = next(message.pandaStates[0] for message in mutable if message.which() == "pandaStates")
  panda_state.canState2.totalRxLostCnt = 1
  save_log(str(segment / "rlog.zst"), [message.as_reader() for message in mutable])

  with pytest.raises(RouteValidationError, match="lost CAN frames"):
    validate_route(segment)


def test_rejects_qcamera_stall_while_main_continues(tmp_path: Path):
  segment = make_segment(tmp_path, "00000000--0123456789", 0)
  messages = [message.as_builder() for message in LogReader(str(segment / "rlog.zst"))]
  ending = messages.pop()
  for offset in range(10):
    messages.append(event("roadCameraState", 260 + offset))
    messages.append(event("roadEncodeIdx", 260 + offset))
  messages.append(ending)
  save_log(str(segment / "rlog.zst"), [message.as_reader() for message in messages])

  with pytest.raises(RouteValidationError, match="qcamera/main frame mismatch"):
    validate_route(segment)


def test_rejects_long_panda_health_gap(tmp_path: Path):
  messages = [event("pandaStates", 1), event("pandaStates", PANDA_STATE_MAX_GAP_NS + 2)]

  with pytest.raises(RouteValidationError, match="health gap exceeds"):
    _require_panda_coverage(tmp_path, messages, (1, PANDA_STATE_MAX_GAP_NS + 2))


def test_rejects_missing_capture_timestamps(tmp_path: Path):
  segment = make_segment(tmp_path, "00000000--0123456789", 0, capture_timestamps=False)

  with pytest.raises(RouteValidationError, match="missing camera monotonic timestamps"):
    validate_route(segment)


def test_rejects_active_segment_by_default(tmp_path: Path):
  segment = make_segment(tmp_path, "00000000--0123456789", 0)
  (segment / "rlog.lock").touch()

  with pytest.raises(RouteValidationError, match="locks present"):
    validate_route(segment)
  assert validate_route(segment, allow_active=True).can_frames == 2


def test_rejects_completed_segment_without_ending_sentinel(tmp_path: Path):
  segment = make_segment(tmp_path, "00000000--0123456789", 0, ending_sentinel=False)

  with pytest.raises(RouteValidationError, match="ending sentinel"):
    validate_route(segment)


def test_log_root_must_contain_one_route(tmp_path: Path):
  make_segment(tmp_path, "00000000--0123456789", 0)
  make_segment(tmp_path, "00000001--9876543210", 0)

  with pytest.raises(RouteValidationError, match="multiple routes"):
    discover_segments(tmp_path)


def test_full_video_probe_decodes_every_frame(tmp_path: Path, monkeypatch):
  video = tmp_path / "fcamera.hevc"
  video.write_bytes(b"video")
  calls = []

  def fake_run(command, **_kwargs):
    calls.append(command)
    if command[0] == "ffprobe":
      return subprocess.CompletedProcess(command, 0, '{"streams":[{"codec_name":"h264","width":1280,"height":720,"nb_read_frames":"20"}]}', "")
    return subprocess.CompletedProcess(command, 0, "", "")

  monkeypatch.setattr(validator, "executable", lambda name: name)
  monkeypatch.setattr(validator.subprocess, "run", fake_run)

  probe = _probe_video(video)

  assert (probe.frames, probe.width, probe.height) == (20, 1280, 720)
  assert any("-xerror" in command for command in calls)


def test_video_probe_rejects_decode_error(tmp_path: Path, monkeypatch):
  video = tmp_path / "fcamera.hevc"
  video.write_bytes(b"corrupt")

  def fake_run(command, **_kwargs):
    if command[0] == "ffprobe":
      return subprocess.CompletedProcess(command, 0, '{"streams":[{"codec_name":"h264","width":1280,"height":720,"nb_read_frames":"20"}]}', "")
    return subprocess.CompletedProcess(command, 1, "", "decode failure")

  monkeypatch.setattr(validator, "executable", lambda name: name)
  monkeypatch.setattr(validator.subprocess, "run", fake_run)

  with pytest.raises(RouteValidationError, match="full H.264 decode failed"):
    _probe_video(video)


def test_video_frame_count_must_match_indexes(tmp_path: Path):
  with pytest.raises(RouteValidationError, match="decoded/indexed frame mismatch"):
    _require_frame_count(tmp_path / "video", decoded_frames=70, indexed_frames=100)
