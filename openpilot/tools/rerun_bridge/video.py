"""Decode and log openpilot camera streams to Rerun."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from openpilot.tools.lib.framereader import FrameReader
from openpilot.tools.lib.route import Route
from openpilot.tools.rerun_bridge.time_axis import ROUTE_TIMELINE

logger = logging.getLogger(__name__)

CAMERA_SPECS = {
  "road": ("fcamera", "roadEncodeIdx"),
  "driver": ("dcamera", "driverEncodeIdx"),
  "wide_road": ("ecamera", "wideRoadEncodeIdx"),
  "q_road": ("qcamera", "qRoadEncodeIdx"),
}

BATCH_SIZE = 64


@dataclass
class CameraFrameRef:
  timestamp: float
  segment: int
  decode_index: int


def _route_camera_paths(route: Route, camera_attr: str) -> dict[int, str]:
  paths: dict[int, str] = {}
  getter = {
    "fcamera": lambda s: s.camera_path,
    "dcamera": lambda s: s.dcamera_path,
    "ecamera": lambda s: s.ecamera_path,
    "qcamera": lambda s: s.qcamera_path,
  }[camera_attr]
  for segment in route.segments:
    path = getter(segment)
    if path:
      paths[segment.name.segment_num] = path
  return paths


def _build_camera_index(series: dict[str, tuple[list[float], list[float]]], encode_idx: str, segments: set[int]) -> list[CameraFrameRef]:
  prefix = f"/{encode_idx}"
  segment_series = series.get(f"{prefix}/segmentNum")
  decode_series = series.get(f"{prefix}/segmentId") or series.get(f"{prefix}/segmentIdEncode")
  if segment_series is None or decode_series is None:
    return []

  seg_t, seg_v = segment_series
  _, decode_v = decode_series
  count = min(len(seg_t), len(decode_v))
  entries: list[CameraFrameRef] = []
  for i in range(count):
    segment = int(round(seg_v[i]))
    if segment not in segments:
      continue
    entries.append(CameraFrameRef(timestamp=seg_t[i], segment=segment, decode_index=int(round(decode_v[i]))))
  entries.sort(key=lambda e: e.timestamp)
  return entries


def _flush_batch(rr, entity: str, times: list[float], frames: list[np.ndarray]) -> int:
  if not times:
    return 0
  rr.send_columns(
    entity,
    indexes=[rr.TimeColumn(ROUTE_TIMELINE, duration=np.asarray(times, dtype=np.float64))],
    columns=rr.Image.columns(buffer=frames),
  )
  count = len(times)
  times.clear()
  frames.clear()
  return count


def log_camera_streams(
  rr,
  route: Route,
  series: dict[str, tuple[list[float], list[float]]],
  frame_skip: int = 1,
) -> dict[str, int]:
  counts: dict[str, int] = {}
  for view_name, (camera_attr, encode_idx) in CAMERA_SPECS.items():
    segment_paths = _route_camera_paths(route, camera_attr)
    if not segment_paths:
      logger.info("camera %s: no segment files", view_name)
      continue

    entries = _build_camera_index(series, encode_idx, set(segment_paths))
    if not entries:
      logger.info("camera %s: no encode index", view_name)
      continue

    readers: dict[int, FrameReader] = {}
    failed_segments: set[int] = set()
    entity = f"camera/{view_name}"
    batch_times: list[float] = []
    batch_frames: list[np.ndarray] = []
    logged = 0

    for i, entry in enumerate(entries):
      if frame_skip > 1 and (i % frame_skip) != 0:
        continue
      if entry.segment in failed_segments:
        continue
      reader = readers.get(entry.segment)
      if reader is None:
        try:
          reader = FrameReader(segment_paths[entry.segment], pix_fmt="rgb24")
          readers[entry.segment] = reader
        except Exception as exc:
          failed_segments.add(entry.segment)
          logger.warning("camera %s segment %s failed to open: %s", view_name, entry.segment, exc)
          continue
      if entry.decode_index < 0 or entry.decode_index >= reader.frame_count:
        continue
      try:
        frame = reader.get(entry.decode_index)
      except Exception:
        continue
      if not isinstance(frame, np.ndarray):
        continue
      batch_times.append(entry.timestamp)
      batch_frames.append(frame)
      if len(batch_frames) >= BATCH_SIZE:
        logged += _flush_batch(rr, entity, batch_times, batch_frames)

    logged += _flush_batch(rr, entity, batch_times, batch_frames)
    counts[view_name] = logged
    logger.info("camera %s: logged %d frames", view_name, logged)
  return counts