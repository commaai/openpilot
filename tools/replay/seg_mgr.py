#!/usr/bin/env python3
import threading
from dataclasses import dataclass, field
from enum import Enum, IntFlag, auto
from typing import Callable, Optional

from openpilot.selfdrive.test.process_replay.migration import migrate_all
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.route import Route, Segment
from openpilot.tools.lib.framereader import FrameReader

MIN_SEGMENTS_CACHE = 5


class ReplayFlags(IntFlag):
  NONE = 0x0000
  DCAM = 0x0002
  ECAM = 0x0004
  NO_LOOP = 0x0010
  NO_FILE_CACHE = 0x0020
  QCAMERA = 0x0040
  NO_VIPC = 0x0400
  ALL_SERVICES = 0x0800


class LoadState(Enum):
  LOADING = auto()
  LOADED = auto()
  FAILED = auto()


@dataclass
class SegmentData:
  seg_num: int
  segment: Segment
  events: list = field(default_factory=list)
  frame_readers: dict = field(default_factory=dict)  # CameraType -> FrameReader
  load_state: LoadState = LoadState.LOADING


@dataclass
class EventData:
  events: list = field(default_factory=list)
  segments: dict = field(default_factory=dict)  # seg_num -> SegmentData

  def is_segment_loaded(self, n: int) -> bool:
    return n in self.segments


class SegmentManager:
  def __init__(self, route_name: str, flags: int = 0, data_dir: str = "", auto_source: bool = False):
    self._flags = flags
    self._data_dir = data_dir if data_dir else None
    self._auto_source = auto_source
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
  def route(self) -> Route:
    return self._route

  def load(self) -> bool:
    try:
      self._route = Route(self._route_name, data_dir=self._data_dir)
    except Exception as e:
      print(f"failed to load route: {self._route_name}: {e}")
      return False

    # Initialize segment slots for all available segments
    for seg in self._route.segments:
      seg_num = seg.name.segment_num
      if seg.log_path or seg.qlog_path:
        self._segments[seg_num] = None

    if not self._segments:
      print(f"no valid segments in route: {self._route_name}")
      return False

    print(f"loaded route {self._route_name} with {len(self._segments)} valid segments")
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

    for seg_num in forward + backward:
      with self._lock:
        if self._exit:
          return loaded_any
        if self._segments.get(seg_num) is not None:
          continue

      # Load segment (blocking - downloads and parses)
      seg_data = self._load_segment(seg_num)
      with self._cv:
        self._segments[seg_num] = seg_data
        self._needs_update = True
        self._cv.notify_all()
      loaded_any = True

      # Only load one segment at a time to be responsive
      return loaded_any

    return loaded_any

  def _load_segment(self, seg_num: int) -> SegmentData:
    # Find the segment object from route
    segment = None
    for seg in self._route.segments:
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
        lr = LogReader(log_path)
        # Apply schema migrations
        events = list(migrate_all(lr))
        seg_data.events = sorted(events, key=lambda x: x.logMonoTime)
      except Exception as e:
        print(f"failed to load log for segment {seg_num}: {e}")
        seg_data.load_state = LoadState.FAILED
        return seg_data

    # Load frame readers based on flags
    try:
      if segment.camera_path and not (self._flags & ReplayFlags.NO_VIPC):
        if not (self._flags & ReplayFlags.QCAMERA):
          seg_data.frame_readers['road'] = FrameReader(segment.camera_path)

      if segment.dcamera_path and (self._flags & ReplayFlags.DCAM):
        seg_data.frame_readers['driver'] = FrameReader(segment.dcamera_path)

      if segment.ecamera_path and (self._flags & ReplayFlags.ECAM):
        seg_data.frame_readers['wide'] = FrameReader(segment.ecamera_path)

      if segment.qcamera_path and (self._flags & ReplayFlags.QCAMERA):
        seg_data.frame_readers['qcam'] = FrameReader(segment.qcamera_path)
    except Exception as e:
      print(f"failed to load frames for segment {seg_num}: {e}")
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

    with self._lock:
      self._event_data = merged_event_data
      self._merged_segments = segments_to_merge

    return True
