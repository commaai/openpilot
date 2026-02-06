#!/usr/bin/env python3
import bz2
import logging
import os
import struct
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, IntFlag, auto
from typing import Optional
from urllib.parse import urlparse

from cereal import messaging, log as capnp_log

from openpilot.tools.lib.framereader import FrameReader
from openpilot.tools.lib.filereader import FileReader
from openpilot.tools.lib.logreader import decompress_stream
from openpilot.tools.lib.route import Route, Segment
from openpilot.tools.replay.persistent_framereader import PersistentFFmpegFrameReader
from openpilot.tools.replay.qcamera_framereader import QCameraFrameReader

log = logging.getLogger("replay")

MIN_SEGMENTS_CACHE = 5
MAX_LOADS_PER_UPDATE = max(1, int(os.getenv("REPLAY_LOADS_PER_UPDATE", "1")))


class ReplayFlags(IntFlag):
  NONE = 0x0000
  DCAM = 0x0002
  ECAM = 0x0004
  NO_LOOP = 0x0010
  QCAMERA = 0x0040
  NO_VIPC = 0x0400
  ALL_SERVICES = 0x0800
  BENCHMARK = 0x1000


class LoadState(Enum):
  LOADING = auto()
  LOADED = auto()
  FAILED = auto()


@dataclass
class SegmentData:
  seg_num: int
  segment: Optional[Segment]
  events: list = field(default_factory=list)
  frame_readers: dict = field(default_factory=dict)  # CameraType -> FrameReader
  load_state: LoadState = LoadState.LOADING


@dataclass
class EventData:
  events: list = field(default_factory=list)
  event_times: list[int] = field(default_factory=list)
  segments: dict = field(default_factory=dict)  # seg_num -> SegmentData

  def is_segment_loaded(self, n: int) -> bool:
    return n in self.segments


@dataclass(slots=True)
class ReplayEvent:
  logMonoTime: int
  which_name: str
  msg_bytes: bytes
  eidx_segnum: int = -1
  frame_segment_id: int = -1
  frame_id: int = 0
  timestamp_sof: int = 0
  timestamp_eof: int = 0

  def which(self) -> str:
    return self.which_name


_U32 = struct.Struct("<I")
_ENCODE_SERVICES = {"roadEncodeIdx", "driverEncodeIdx", "wideRoadEncodeIdx"}


def _iter_serialized_messages(data: bytes):
  mv = memoryview(data)
  offset = 0
  total = len(mv)

  while offset + 8 <= total:
    seg_count = _U32.unpack_from(mv, offset)[0] + 1
    table_words = 1 + seg_count
    table_bytes = table_words * 4
    if table_words % 2:
      table_bytes += 4

    if offset + table_bytes > total:
      break

    sizes_offset = offset + 4
    payload_words = 0
    for _ in range(seg_count):
      payload_words += _U32.unpack_from(mv, sizes_offset)[0]
      sizes_offset += 4

    msg_end = offset + table_bytes + payload_words * 8
    if msg_end <= offset or msg_end > total:
      break

    yield mv[offset:msg_end]
    offset = msg_end


def _read_log_data(log_path: str) -> bytes:
  with FileReader(log_path) as f:
    data = f.read()

  if not data:
    return b""

  ext = os.path.splitext(urlparse(log_path).path)[1]
  if ext == ".bz2" or data.startswith(b"BZh9"):
    return bz2.decompress(data)
  if ext == ".zst" or data.startswith(b"\x28\xb5\x2f\xfd"):
    return decompress_stream(data)
  return data


def _migrate_old_controls_state(events: list[ReplayEvent]) -> None:
  if any(evt.which_name == "selfdriveState" for evt in events):
    return

  migrated = []
  fields = (
    ("active", "activeDEPRECATED"),
    ("alertSize", "alertSizeDEPRECATED"),
    ("alertStatus", "alertStatusDEPRECATED"),
    ("alertText1", "alertText1DEPRECATED"),
    ("alertText2", "alertText2DEPRECATED"),
    ("alertType", "alertTypeDEPRECATED"),
    ("enabled", "enabledDEPRECATED"),
    ("engageable", "engageableDEPRECATED"),
    ("experimentalMode", "experimentalModeDEPRECATED"),
    ("personality", "personalityDEPRECATED"),
    ("state", "stateDEPRECATED"),
  )

  for evt in events:
    if evt.which_name != "controlsState":
      continue

    with capnp_log.Event.from_bytes(evt.msg_bytes) as old_evt:
      cs = old_evt.controlsState

      new_evt = messaging.new_message("selfdriveState", valid=old_evt.valid, logMonoTime=old_evt.logMonoTime)
      ss = new_evt.selfdriveState

      for new_field, old_field in fields:
        setattr(ss, new_field, getattr(cs, old_field))

      try:
        ss.alertSound = cs.alertSound2DEPRECATED
      except Exception:
        pass

      migrated.append(
        ReplayEvent(
          logMonoTime=old_evt.logMonoTime,
          which_name="selfdriveState",
          msg_bytes=new_evt.to_bytes(),
        )
      )

  events.extend(migrated)


def _parse_replay_events(log_path: str, include_frame_events: bool) -> list[ReplayEvent]:
  data = _read_log_data(log_path)
  if not data:
    return []

  events = []
  for msg_view in _iter_serialized_messages(data):
    msg_bytes = bytes(msg_view)

    try:
      with capnp_log.Event.from_bytes(msg_bytes) as evt:
        which = evt.which()
        mono_time = evt.logMonoTime

        events.append(
          ReplayEvent(
            logMonoTime=mono_time,
            which_name=which,
            msg_bytes=msg_bytes,
          )
        )

        if include_frame_events and which in _ENCODE_SERVICES:
          eidx = getattr(evt, which)
          if str(eidx.type) == "fullHEVC":
            sof = eidx.timestampSof
            events.append(
              ReplayEvent(
                logMonoTime=sof if sof else mono_time,
                which_name=which,
                msg_bytes=msg_bytes,
                eidx_segnum=eidx.segmentNum,
                frame_segment_id=eidx.segmentId,
                frame_id=eidx.frameId,
                timestamp_sof=sof,
                timestamp_eof=eidx.timestampEof,
              )
            )
    except Exception:
      log.debug("failed to parse event", exc_info=True)

  _migrate_old_controls_state(events)
  events.sort(key=lambda e: (e.logMonoTime, e.which_name))
  return events


def _extract_car_params(log_path: str) -> Optional[tuple[str, bytes]]:
  data = _read_log_data(log_path)
  if not data:
    return None

  for msg_view in _iter_serialized_messages(data):
    try:
      with capnp_log.Event.from_bytes(bytes(msg_view)) as evt:
        if evt.which() != "carParams":
          continue
        car_params = evt.carParams
        return car_params.carFingerprint, car_params.as_builder().to_bytes()
    except Exception:
      continue

  return None


def _expected_frame_count(events: list[ReplayEvent], seg_num: int, service: str) -> Optional[int]:
  max_idx = -1
  for evt in events:
    if evt.which_name == service and evt.eidx_segnum == seg_num and evt.frame_segment_id >= 0:
      if evt.frame_segment_id > max_idx:
        max_idx = evt.frame_segment_id
  return (max_idx + 1) if max_idx >= 0 else None


def _start_reader_prefetch(reader, initial_frames: int = 48) -> None:
  if not hasattr(reader, "prefetch_to"):
    return
  try:
    target = min(reader.frame_count - 1, max(0, initial_frames - 1))
    if target >= 0:
      reader.prefetch_to(target)
  except Exception:
    log.debug("failed to start reader prefetch", exc_info=True)


class SegmentManager:
  def __init__(self, route_name: str, flags: int = 0, data_dir: str = ""):
    self._flags = flags
    self._data_dir = data_dir if data_dir else None
    self._route_name = route_name
    self._route: Optional[Route] = None

    self._filters: list[bool] = []
    self._segments: dict[int, Optional[SegmentData]] = {}
    self._event_data = EventData()
    self._merged_segments: set[int] = set()

    self._lock = threading.Lock()
    self._cv = threading.Condition(self._lock)
    self._thread: Optional[threading.Thread] = None
    self._cur_seg_num = -1
    self._needs_update = False
    self._exit = False
    self._initial_car_params: Optional[tuple[str, bytes]] = None

    self.segment_cache_limit = MIN_SEGMENTS_CACHE
    self._on_segment_merged_callback: Optional[Callable[[], None]] = None

  def __del__(self):
    if hasattr(self, '_cv'):
      with self._cv:
        self._exit = True
        self._cv.notify_all()
    if hasattr(self, '_thread') and self._thread is not None and self._thread.is_alive():
      self._thread.join()

  @property
  def route(self) -> Optional[Route]:
    return self._route

  @property
  def initial_car_params(self) -> Optional[tuple[str, bytes]]:
    return self._initial_car_params

  def load(self) -> bool:
    try:
      self._route = Route(self._route_name, data_dir=self._data_dir)
    except Exception:
      log.exception(f"failed to load route: {self._route_name}")
      return False

    # Initialize segment slots for all available segments
    for seg in self._route.segments:
      seg_num = seg.name.segment_num
      if seg.log_path or seg.qlog_path:
        self._segments[seg_num] = None

    if not self._segments:
      log.error(f"no valid segments in route: {self._route_name}")
      return False

    if not (self._flags & ReplayFlags.BENCHMARK):
      # Fast-path startup metadata from the first segment using qlog when available
      first_seg_num = min(self._segments.keys())
      first_seg = next((s for s in self._route.segments if s.name.segment_num == first_seg_num), None)
      if first_seg is not None:
        for path in (first_seg.qlog_path, first_seg.log_path):
          if not path:
            continue
          try:
            self._initial_car_params = _extract_car_params(path)
          except Exception:
            self._initial_car_params = None
          if self._initial_car_params is not None:
            break

    log.info(f"loaded route {self._route_name} with {len(self._segments)} valid segments")
    self._thread = threading.Thread(target=self._manage_segment_cache, daemon=True)
    self._thread.start()
    return True

  def set_current_segment(self, seg_num: int) -> None:
    with self._cv:
      if self._cur_seg_num == seg_num:
        return
      self._cur_seg_num = seg_num
      self._needs_update = True
      self._cv.notify_all()

  def set_callback(self, callback: Callable[[], None]) -> None:
    self._on_segment_merged_callback = callback

  def set_filters(self, filters: list[bool]) -> None:
    self._filters = filters

  def get_event_data(self) -> EventData:
    with self._lock:
      return self._event_data

  def has_segment(self, n: int) -> bool:
    return n in self._segments

  def _manage_segment_cache(self) -> None:
    while True:
      with self._cv:
        self._cv.wait_for(lambda: self._exit or self._needs_update)
        if self._exit:
          break

        self._needs_update = False
        seg_nums = sorted(self._segments.keys())
        if not seg_nums:
          continue

        # Find current segment index
        cur_idx = 0
        for i, n in enumerate(seg_nums):
          if n >= self._cur_seg_num:
            cur_idx = i
            break

        # Calculate range to load
        half_cache = self.segment_cache_limit // 2
        begin_idx = max(0, cur_idx - half_cache)
        end_idx = min(len(seg_nums), begin_idx + self.segment_cache_limit)
        begin_idx = max(0, end_idx - self.segment_cache_limit)

        range_seg_nums = seg_nums[begin_idx:end_idx]

      # Load segments in range (outside lock)
      self._load_segments_in_range(range_seg_nums, self._cur_seg_num)
      merged = self._merge_segments(range_seg_nums)

      # Free segments outside range
      with self._lock:
        for seg_num in list(self._segments.keys()):
          if seg_num not in range_seg_nums:
            self._segments[seg_num] = None

      if merged and self._on_segment_merged_callback:
        self._on_segment_merged_callback()

  def _load_segments_in_range(self, seg_nums: list[int], cur_seg_num: int) -> bool:
    """Load segments in range. Returns True if any segment was loaded."""
    # Load forward from current, then backward
    forward = [n for n in seg_nums if n >= cur_seg_num]
    backward = [n for n in seg_nums if n < cur_seg_num][::-1]
    loaded_any = False
    loaded_count = 0

    for seg_num in forward + backward:
      with self._lock:
        if self._exit:
          return loaded_any
        if self._segments.get(seg_num) is not None:
          continue

      # Load segment (blocking - downloads and parses)
      log.info(f"loading segment {seg_num}...")
      seg_data = self._load_segment(seg_num)
      with self._cv:
        self._segments[seg_num] = seg_data
        self._needs_update = True
        self._cv.notify_all()
      loaded_any = True
      loaded_count += 1
      log.info(f"segment {seg_num} loaded with {len(seg_data.events)} events")

      if loaded_count >= MAX_LOADS_PER_UPDATE:
        return loaded_any

    return loaded_any

  def _load_segment(self, seg_num: int) -> SegmentData:
    # Find the segment object from route
    route = self._route
    if route is None:
      return SegmentData(seg_num=seg_num, segment=None, load_state=LoadState.FAILED)

    segment = None
    for seg in route.segments:
      if seg.name.segment_num == seg_num:
        segment = seg
        break

    if segment is None:
      return SegmentData(seg_num=seg_num, segment=None, load_state=LoadState.FAILED)

    seg_data = SegmentData(seg_num=seg_num, segment=segment)

    # Load log events
    log_path = segment.log_path or segment.qlog_path
    if log_path:
      try:
        include_frame_events = not bool(self._flags & ReplayFlags.NO_VIPC)
        seg_data.events = _parse_replay_events(log_path, include_frame_events)
      except Exception as e:
        log.warning(f"failed to load log for segment {seg_num}: {e}")
        seg_data.load_state = LoadState.FAILED
        return seg_data

    # Load frame readers based on flags
    # VisionIPC expects NV12 format
    try:
      # cache_size=90 holds 3 GOPs (30 frames each) to stay ahead at 8x speed
      use_persistent_decoder = os.getenv("REPLAY_PERSISTENT_DECODER", "1") != "0"
      road_frame_count = _expected_frame_count(seg_data.events, seg_num, "roadEncodeIdx")
      driver_frame_count = _expected_frame_count(seg_data.events, seg_num, "driverEncodeIdx")
      wide_frame_count = _expected_frame_count(seg_data.events, seg_num, "wideRoadEncodeIdx")

      if not (self._flags & ReplayFlags.NO_VIPC):
        use_qcamera = bool(self._flags & ReplayFlags.QCAMERA)
        road_camera_path = segment.qcamera_path if (use_qcamera or not segment.camera_path) else segment.camera_path
        if road_camera_path:
          road_name = os.path.basename(urlparse(road_camera_path).path)
          if road_name == "qcamera.ts":
            seg_data.frame_readers['road'] = QCameraFrameReader(road_camera_path, cache_size=90)
          elif use_persistent_decoder:
            seg_data.frame_readers['road'] = PersistentFFmpegFrameReader(road_camera_path, pix_fmt='nv12', cache_size=90, expected_frame_count=road_frame_count)
            _start_reader_prefetch(seg_data.frame_readers['road'])
          else:
            seg_data.frame_readers['road'] = FrameReader(road_camera_path, pix_fmt='nv12', cache_size=90)

      if segment.dcamera_path and (self._flags & ReplayFlags.DCAM):
        if use_persistent_decoder:
          seg_data.frame_readers['driver'] = PersistentFFmpegFrameReader(
            segment.dcamera_path, pix_fmt='nv12', cache_size=90, expected_frame_count=driver_frame_count
          )
          _start_reader_prefetch(seg_data.frame_readers['driver'])
        else:
          seg_data.frame_readers['driver'] = FrameReader(segment.dcamera_path, pix_fmt='nv12', cache_size=90)

      if segment.ecamera_path and (self._flags & ReplayFlags.ECAM):
        if use_persistent_decoder:
          seg_data.frame_readers['wide'] = PersistentFFmpegFrameReader(
            segment.ecamera_path, pix_fmt='nv12', cache_size=90, expected_frame_count=wide_frame_count
          )
          _start_reader_prefetch(seg_data.frame_readers['wide'])
        else:
          seg_data.frame_readers['wide'] = FrameReader(segment.ecamera_path, pix_fmt='nv12', cache_size=90)

    except Exception as e:
      log.warning(f"failed to load frames for segment {seg_num}: {e}")
      # Don't fail the whole segment, just skip frames

    seg_data.load_state = LoadState.LOADED
    return seg_data

  def _merge_segments(self, seg_nums: list[int]) -> bool:
    segments_to_merge = set()
    total_events = []

    with self._lock:
      for seg_num in seg_nums:
        seg_data = self._segments.get(seg_num)
        if seg_data and seg_data.load_state == LoadState.LOADED:
          segments_to_merge.add(seg_num)

      if segments_to_merge == self._merged_segments:
        return False

    # Merge events from all loaded segments
    merged_event_data = EventData()
    for seg_num in sorted(segments_to_merge):
      with self._lock:
        seg_data = self._segments.get(seg_num)
        if seg_data is None:
          continue

      events = seg_data.events
      if not events:
        continue

      # Skip initData if present (first event)
      start_idx = 0
      if events and events[0].which() == 'initData':
        start_idx = 1

      total_events.extend(events[start_idx:])
      merged_event_data.segments[seg_num] = seg_data

    # Sort all events by time
    merged_event_data.events = sorted(total_events, key=lambda x: x.logMonoTime)
    merged_event_data.event_times = [evt.logMonoTime for evt in merged_event_data.events]

    with self._lock:
      self._event_data = merged_event_data
      self._merged_segments = segments_to_merge

    log.debug(f"merged segments: {sorted(segments_to_merge)}")
    return True
