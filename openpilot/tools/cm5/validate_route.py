#!/usr/bin/env python3
"""Validate locally recorded openpilot camera and CAN route segments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import pairwise
import json
from pathlib import Path
import re
import subprocess
import sys

from openpilot.cereal import log
from openpilot.tools.cm5.ffmpeg import executable
from openpilot.tools.cm5.usb_pandad import parse_bus_indices
from openpilot.tools.lib.logreader import LogReader


SEGMENT_RE = re.compile(r"^(?P<prefix>.+)--(?P<segment>[0-9]+)$")
REQUIRED_FILES = ("rlog.zst", "qlog.zst", "fcamera.hevc", "qcamera.ts")
PANDA_STATE_MAX_GAP_NS = 2_000_000_000
VIDEO_COVERAGE_TOLERANCE_NS = 500_000_000


class RouteValidationError(Exception):
  pass


@dataclass(frozen=True)
class VideoProbe:
  frames: int
  width: int
  height: int


@dataclass(frozen=True)
class SegmentStats:
  path: Path
  number: int
  can_events: int
  can_frames: int
  camera_states: int
  encode_indexes: int
  video_bytes: int
  qvideo_bytes: int


@dataclass(frozen=True)
class RouteStats:
  prefix: Path
  segments: tuple[SegmentStats, ...]

  @property
  def can_frames(self) -> int:
    return sum(segment.can_frames for segment in self.segments)

  @property
  def encoded_frames(self) -> int:
    return sum(segment.encode_indexes for segment in self.segments)

  @property
  def trimmed(self) -> bool:
    return self.segments[0].number != 0


def _segment_parts(path: Path) -> tuple[str, int] | None:
  match = SEGMENT_RE.fullmatch(path.name)
  if match is None:
    return None
  return match.group("prefix"), int(match.group("segment"))


def discover_segments(target: Path) -> tuple[Path, list[tuple[int, Path]]]:
  """Resolve a segment, route prefix, or single-route log root."""
  target = target.expanduser().resolve()
  candidates: list[Path]
  selected_prefix: str | None = None

  if target.is_dir() and _segment_parts(target) is not None:
    parts = _segment_parts(target)
    if parts is None:
      raise RouteValidationError(f"segment directory has no numeric suffix: {target}")
    selected_prefix = parts[0]
    candidates = [path for path in target.parent.iterdir() if path.is_dir()]
  elif target.is_dir():
    candidates = [path for path in target.iterdir() if path.is_dir() and _segment_parts(path) is not None]
  else:
    selected_prefix = target.name
    candidates = [path for path in target.parent.glob(f"{target.name}--*") if path.is_dir()]

  grouped: dict[str, list[tuple[int, Path]]] = {}
  for path in candidates:
    parts = _segment_parts(path)
    if parts is None:
      continue
    prefix, segment = parts
    if selected_prefix is None or prefix == selected_prefix:
      grouped.setdefault(prefix, []).append((segment, path))

  if not grouped:
    raise RouteValidationError(f"no route segments found for {target}")
  if len(grouped) != 1:
    prefixes = ", ".join(sorted(grouped))
    raise RouteValidationError(f"multiple routes found ({prefixes}); pass one segment directory or an exact route prefix")

  prefix_name, segments = next(iter(grouped.items()))
  segments.sort(key=lambda item: item[0])
  return segments[0][1].parent / prefix_name, segments


def _probe_video(path: Path) -> VideoProbe:
  ffprobe = executable("ffprobe")
  ffmpeg = executable("ffmpeg")
  if ffprobe is None or ffmpeg is None:
    raise RouteValidationError("video decoding was requested but ffprobe/ffmpeg is not installed")
  command = [
    ffprobe,
    "-v",
    "error",
    "-select_streams",
    "v:0",
    "-count_frames",
    "-show_entries",
    "stream=codec_name,width,height,nb_read_frames",
    "-of",
    "json",
    str(path),
  ]
  try:
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=120)
  except subprocess.TimeoutExpired as exc:
    raise RouteValidationError(f"ffprobe timed out for {path}") from exc
  if result.returncode != 0:
    raise RouteValidationError(f"ffprobe failed for {path}: {result.stderr.strip()}")
  try:
    streams = json.loads(result.stdout)["streams"]
  except (KeyError, json.JSONDecodeError) as exc:
    raise RouteValidationError(f"invalid ffprobe output for {path}") from exc
  try:
    stream = streams[0] if streams else {}
    frame_count = int(stream.get("nb_read_frames", 0))
    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))
  except (TypeError, ValueError) as exc:
    raise RouteValidationError(f"invalid video metadata from ffprobe for {path}") from exc
  if stream.get("codec_name") != "h264":
    raise RouteValidationError(f"expected H.264 video in {path}, found {stream.get('codec_name', 'no codec')}")
  if width <= 0 or height <= 0:
    raise RouteValidationError(f"invalid video dimensions in {path}: {width}x{height}")
  if frame_count < 1:
    raise RouteValidationError(f"no decodable video frames in {path}")

  decode_command = [
    ffmpeg, "-nostdin", "-v", "error", "-xerror", "-i", str(path),
    "-map", "0:v:0", "-c:v", "rawvideo", "-f", "null", "-",
  ]
  try:
    decoded = subprocess.run(decode_command, capture_output=True, text=True, check=False, timeout=120)
  except subprocess.TimeoutExpired as exc:
    raise RouteValidationError(f"video decode timed out for {path}") from exc
  if decoded.returncode != 0:
    detail = decoded.stderr.strip() or f"exit status {decoded.returncode}"
    raise RouteValidationError(f"full H.264 decode failed for {path}: {detail}")
  return VideoProbe(frames=frame_count, width=width, height=height)


def _require_frame_count(path: Path, decoded_frames: int, indexed_frames: int) -> None:
  tolerance = max(2, round(indexed_frames * 0.02))
  if abs(decoded_frames - indexed_frames) > tolerance:
    raise RouteValidationError(
      f"{path}: decoded/indexed frame mismatch ({decoded_frames} decoded, {indexed_frames} indexed, tolerance {tolerance})"
    )


def _timestamp_range(messages, field: str, *, description: str) -> tuple[int, int]:
  timestamps = [getattr(message, field) for message in messages if getattr(message, field) > 0]
  if not timestamps:
    raise RouteValidationError(f"missing {description} monotonic timestamps")
  return min(timestamps), max(timestamps)


def _require_overlap(path: Path, can_range: tuple[int, int], capture_range: tuple[int, int], description: str) -> None:
  if max(can_range[0], capture_range[0]) > min(can_range[1], capture_range[1]):
    raise RouteValidationError(f"{path}: CAN and {description} timestamps do not overlap")


def _require_ending_sentinel(messages, path: Path) -> None:
  ending_types = (log.Sentinel.SentinelType.endOfSegment, log.Sentinel.SentinelType.endOfRoute)
  if not messages or messages[-1].which() != "sentinel" or messages[-1].sentinel.type not in ending_types:
    raise RouteValidationError(f"{path}: log has no valid ending sentinel")


def _require_panda_coverage(path: Path, messages, recording_range: tuple[int, int]) -> None:
  timestamps = sorted(message.logMonoTime for message in messages if message.logMonoTime > 0)
  if not timestamps:
    raise RouteValidationError(f"{path}: Panda health has no monotonic timestamps")
  if timestamps[0] > recording_range[0] + PANDA_STATE_MAX_GAP_NS or timestamps[-1] + PANDA_STATE_MAX_GAP_NS < recording_range[1]:
    raise RouteValidationError(f"{path}: Panda health does not cover the camera recording")
  if any(end - start > PANDA_STATE_MAX_GAP_NS for start, end in pairwise(timestamps)):
    raise RouteValidationError(f"{path}: Panda health gap exceeds {PANDA_STATE_MAX_GAP_NS / 1e9:g} seconds")


def _require_can_bus_coverage(path: Path, messages, recording_range: tuple[int, int],
                              required_buses: frozenset[int], max_gap_ns: int) -> None:
  for bus in sorted(required_buses):
    timestamps = sorted(
      message.logMonoTime for message in messages
      if message.logMonoTime > 0 and any(frame.src == bus for frame in message.can)
    )
    if not timestamps:
      raise RouteValidationError(f"{path}: required CAN bus {bus} has no frames")
    if timestamps[0] > recording_range[0] + max_gap_ns or timestamps[-1] + max_gap_ns < recording_range[1]:
      raise RouteValidationError(f"{path}: required CAN bus {bus} does not cover the recording")
    if any(end - start > max_gap_ns for start, end in pairwise(timestamps)):
      raise RouteValidationError(f"{path}: required CAN bus {bus} gap exceeds {max_gap_ns / 1e9:g} seconds")


def validate_segment(path: Path, number: int, *, allow_active: bool = False, ffprobe: bool = False,
                     required_can_buses: frozenset[int] = frozenset(), max_can_gap_ns: int = 5_000_000_000) -> SegmentStats:
  missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
  if missing:
    raise RouteValidationError(f"{path}: missing {', '.join(missing)}")
  empty = [name for name in REQUIRED_FILES if (path / name).stat().st_size == 0]
  if empty:
    raise RouteValidationError(f"{path}: empty {', '.join(empty)}")

  locks = sorted(path.glob("*.lock"))
  if locks and not allow_active:
    raise RouteValidationError(f"{path}: active/incomplete segment locks present: {', '.join(lock.name for lock in locks)}")

  try:
    messages = list(LogReader(str(path / "rlog.zst")))
    # Parsing qlog catches truncation even though counts come from the full log.
    qmessages = list(LogReader(str(path / "qlog.zst")))
  except Exception as exc:
    raise RouteValidationError(f"{path}: unreadable log: {exc}") from exc

  if not locks:
    _require_ending_sentinel(messages, path / "rlog.zst")
    _require_ending_sentinel(qmessages, path / "qlog.zst")

  types = [message.which() for message in messages]
  if "initData" not in types:
    raise RouteValidationError(f"{path}: rlog has no initData")
  init_data = next(message.initData for message in messages if message.which() == "initData")
  if not init_data.passive:
    raise RouteValidationError(f"{path}: initData is not marked passive")

  can_messages = [message for message in messages if message.which() == "can"]
  can_frames = sum(len(message.can) for message in can_messages)
  camera_messages = [message for message in messages if message.which() == "roadCameraState"]
  encode_messages = [message for message in messages if message.which() == "roadEncodeIdx"]
  qencode_messages = [message for message in messages if message.which() == "qRoadEncodeIdx"]
  panda_messages = [message for message in messages if message.which() == "pandaStates"]

  if can_frames == 0:
    raise RouteValidationError(f"{path}: rlog has no CAN frames")
  if not camera_messages:
    raise RouteValidationError(f"{path}: rlog has no roadCameraState messages")
  if not encode_messages:
    raise RouteValidationError(f"{path}: rlog has no roadEncodeIdx messages")
  if any(message.roadEncodeIdx.type != log.EncodeIndex.Type.fullH264 for message in encode_messages):
    raise RouteValidationError(f"{path}: road video is not indexed as fullH264")
  if not qencode_messages or any(message.qRoadEncodeIdx.type != log.EncodeIndex.Type.qcameraH264 for message in qencode_messages):
    raise RouteValidationError(f"{path}: qcamera video has no valid H.264 indexes")
  camera_encode_tolerance = max(2, round(len(camera_messages) * 0.05))
  if abs(len(camera_messages) - len(encode_messages)) > camera_encode_tolerance:
    raise RouteValidationError(
      f"{path}: camera/encoder frame mismatch ({len(camera_messages)} captured, {len(encode_messages)} indexed)"
    )
  qcamera_tolerance = max(2, round(len(encode_messages) * 0.05))
  if abs(len(qencode_messages) - len(encode_messages)) > qcamera_tolerance:
    raise RouteValidationError(
      f"{path}: qcamera/main frame mismatch ({len(qencode_messages)} qcamera, {len(encode_messages)} main)"
    )
  if not panda_messages:
    raise RouteValidationError(f"{path}: rlog has no Panda health state")
  panda_states = [state for message in panda_messages for state in message.pandaStates]
  if not panda_states or any(
    state.pandaType != "redPanda" or state.safetyModel != "silent" or state.controlsAllowed
    for state in panda_states
  ):
    raise RouteValidationError(f"{path}: Panda was not continuously in SILENT receive-only mode")
  if any(
    can_state.totalTxCnt != 0 or can_state.totalFwdCnt != 0
    for state in panda_states for can_state in (state.canState0, state.canState1, state.canState2)
  ):
    raise RouteValidationError(f"{path}: Panda transmitted or forwarded CAN frames")
  if any(state.rxBufferOverflow != 0 for state in panda_states) or any(
    can_state.totalRxLostCnt != 0
    for state in panda_states for can_state in (state.canState0, state.canState1, state.canState2)
  ):
    raise RouteValidationError(f"{path}: Panda reported lost CAN frames")

  can_range = _timestamp_range(can_messages, "logMonoTime", description="CAN")
  camera_range = _timestamp_range((message.roadCameraState for message in camera_messages), "timestampEof", description="camera")
  encode_range = _timestamp_range((message.roadEncodeIdx for message in encode_messages), "timestampEof", description="encoder")
  qencode_range = _timestamp_range((message.qRoadEncodeIdx for message in qencode_messages), "timestampEof", description="qcamera encoder")
  recording_range = (min(camera_range[0], encode_range[0]), max(camera_range[1], encode_range[1]))
  if (qencode_range[0] > recording_range[0] + VIDEO_COVERAGE_TOLERANCE_NS or
      qencode_range[1] + VIDEO_COVERAGE_TOLERANCE_NS < recording_range[1]):
    raise RouteValidationError(f"{path}: qcamera indexes do not cover the main recording")
  _require_panda_coverage(path, panda_messages, recording_range)
  _require_can_bus_coverage(path, can_messages, recording_range, required_can_buses, max_can_gap_ns)
  _require_overlap(path, can_range, camera_range, "camera capture")
  _require_overlap(path, can_range, encode_range, "encoded frame")

  if ffprobe:
    main_probe = _probe_video(path / "fcamera.hevc")
    qcam_probe = _probe_video(path / "qcamera.ts")
    _require_frame_count(path / "fcamera.hevc", main_probe.frames, len(encode_messages))
    _require_frame_count(path / "qcamera.ts", qcam_probe.frames, len(qencode_messages))

  return SegmentStats(
    path=path,
    number=number,
    can_events=len(can_messages),
    can_frames=can_frames,
    camera_states=len(camera_messages),
    encode_indexes=len(encode_messages),
    video_bytes=(path / "fcamera.hevc").stat().st_size,
    qvideo_bytes=(path / "qcamera.ts").stat().st_size,
  )


def validate_route(target: Path, *, allow_active: bool = False, ffprobe: bool = False,
                   required_can_buses: frozenset[int] = frozenset(), max_can_gap_ns: int = 5_000_000_000) -> RouteStats:
  prefix, discovered = discover_segments(target)
  numbers = [number for number, _ in discovered]
  if numbers != list(range(numbers[0], numbers[-1] + 1)):
    raise RouteValidationError(f"{prefix}: available segment numbers are not contiguous: {numbers}")
  segments = tuple(
    validate_segment(
      path, number, allow_active=allow_active, ffprobe=ffprobe,
      required_can_buses=required_can_buses, max_can_gap_ns=max_can_gap_ns,
    )
    for number, path in discovered
  )
  return RouteStats(prefix=prefix, segments=segments)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Validate an openpilot CM5 dashcam route")
  parser.add_argument("target", type=Path, help="segment directory, exact route prefix, or a log root containing one route")
  parser.add_argument("--allow-active", action="store_true", help="allow lock files in an actively recorded segment")
  parser.add_argument(
    "--required-can-buses", type=parse_bus_indices, default=frozenset(),
    help="comma-separated bus indexes that must cover every segment (default: none)",
  )
  parser.add_argument("--max-can-gap", type=float, default=5.0, help="maximum allowed silence on a required CAN bus in seconds")
  parser.add_argument(
    "--decode-video", "--ffprobe", dest="ffprobe", action="store_true",
    help="fully decode both videos and compare decoded frames with their log indexes",
  )
  return parser


def main() -> int:
  args = build_parser().parse_args()
  try:
    if args.max_can_gap <= 0:
      raise RouteValidationError("--max-can-gap must be greater than zero")
    report = validate_route(
      args.target, allow_active=args.allow_active, ffprobe=args.ffprobe,
      required_can_buses=args.required_can_buses, max_can_gap_ns=round(args.max_can_gap * 1e9),
    )
  except RouteValidationError as exc:
    print(f"INVALID: {exc}", file=sys.stderr)
    return 1

  retained_range = f"{report.segments[0].number}-{report.segments[-1].number}"
  trimmed = ", trimmed by retention" if report.trimmed else ""
  summary = f"VALID: {report.prefix.name}: {len(report.segments)} segment(s) [{retained_range}]{trimmed}, "
  summary += f"{report.can_frames} CAN frame(s), {report.encoded_frames} encoded road frame(s)"
  print(summary)
  for segment in report.segments:
    details = f"  {segment.number}: CAN={segment.can_frames} camera={segment.camera_states} encoded={segment.encode_indexes} "
    details += f"video={segment.video_bytes + segment.qvideo_bytes} bytes"
    print(details)
  return 0


if __name__ == "__main__":
  sys.exit(main())
