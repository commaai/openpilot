#!/usr/bin/env python3
import threading
import time
from enum import IntFlag
from typing import Callable, Optional

import cereal.messaging as messaging
from cereal import log as capnp_log
from cereal.services import SERVICE_LIST
from openpilot.common.params import Params

from openpilot.tools.replay.camera import CameraServer, CameraType
from openpilot.tools.replay.seg_mgr import SegmentManager, ReplayFlags
from openpilot.tools.replay.timeline import Timeline, FindFlag

DEMO_ROUTE = "a2a0ccea32023010|2023-07-27--13-01-19"


class Replay:
  def __init__(self, route: str, allow: list[str] = None, block: list[str] = None,
               sm=None, flags: int = 0, data_dir: str = "", auto_source: bool = False):
    self._sm = sm
    self._flags = flags
    self._seg_mgr = SegmentManager(route, flags, data_dir, auto_source)

    allow = allow or []
    block = block or []

    if not (flags & ReplayFlags.ALL_SERVICES):
      block.extend(["bookmarkButton", "uiDebug", "userBookmark"])

    self._sockets: dict[str, bool] = {}  # service name -> enabled
    self._pm: Optional[messaging.PubMaster] = None
    self._setup_services(allow, block)
    self._setup_segment_manager(bool(allow) or bool(block))

    self._timeline = Timeline()
    self._camera_server: Optional[CameraServer] = None

    # Stream thread state
    self._stream_thread: Optional[threading.Thread] = None
    self._stream_lock = threading.Lock()
    self._stream_cv = threading.Condition(self._stream_lock)
    self._interrupt = threading.Event()

    self._user_paused = False
    self._events_ready = False
    self._exit = False

    self._current_segment = 0
    self._seeking_to = -1.0
    self._route_start_ts = 0
    self._cur_mono_time = 0
    self._route_date_time = 0
    self._min_seconds = 0.0
    self._max_seconds = 0.0
    self._speed = 1.0
    self._car_fingerprint = ""

    # Callbacks
    self.on_segments_merged: Optional[Callable[[], None]] = None
    self.on_seeking: Optional[Callable[[float], None]] = None
    self.on_seeked_to: Optional[Callable[[float], None]] = None
    self.on_qlog_loaded: Optional[Callable] = None
    self._event_filter: Optional[Callable] = None

  def __del__(self):
    if hasattr(self, '_stream_thread') and self._stream_thread is not None and self._stream_thread.is_alive():
      print("shutdown: in progress...")
      self._interrupt_stream(lambda: setattr(self, '_exit', True) or False)
      self._stream_thread.join()
      print("shutdown: done")

  def _setup_services(self, allow: list[str], block: list[str]) -> None:
    active_services = []

    for name in SERVICE_LIST.keys():
      is_blocked = name in block
      is_allowed = not allow or name in allow
      if is_allowed and not is_blocked:
        self._sockets[name] = True
        active_services.append(name)
      else:
        self._sockets[name] = False

    print(f"active services: {', '.join(active_services)}")
    if not self._sm:
      self._pm = messaging.PubMaster(active_services)

  def _setup_segment_manager(self, has_filters: bool) -> None:
    self._seg_mgr.set_callback(self._handle_segment_merge)

  @property
  def route(self):
    return self._seg_mgr.route

  @property
  def is_paused(self) -> bool:
    return self._user_paused

  @property
  def current_seconds(self) -> float:
    return (self._cur_mono_time - self._route_start_ts) / 1e9

  @property
  def route_start_nanos(self) -> int:
    return self._route_start_ts

  @property
  def min_seconds(self) -> float:
    return self._min_seconds

  @property
  def max_seconds(self) -> float:
    return self._max_seconds

  @property
  def speed(self) -> float:
    return self._speed

  @speed.setter
  def speed(self, value: float) -> None:
    self._speed = max(0.1, min(8.0, value))

  @property
  def car_fingerprint(self) -> str:
    return self._car_fingerprint

  @property
  def segment_cache_limit(self) -> int:
    return self._seg_mgr.segment_cache_limit

  @segment_cache_limit.setter
  def segment_cache_limit(self, value: int) -> None:
    self._seg_mgr.segment_cache_limit = max(5, value)

  def has_flag(self, flag: ReplayFlags) -> bool:
    return bool(self._flags & flag)

  def set_loop(self, loop: bool) -> None:
    if loop:
      self._flags &= ~ReplayFlags.NO_LOOP
    else:
      self._flags |= ReplayFlags.NO_LOOP

  def loop(self) -> bool:
    return not (self._flags & ReplayFlags.NO_LOOP)

  def get_timeline(self):
    return self._timeline.get_entries()

  def find_alert_at_time(self, sec: float):
    return self._timeline.find_alert_at_time(sec)

  def get_event_data(self):
    return self._seg_mgr.get_event_data()

  def install_event_filter(self, filter_fn: Callable) -> None:
    self._event_filter = filter_fn

  def load(self) -> bool:
    print(f"loading route {self._seg_mgr._route_name}")
    if not self._seg_mgr.load():
      return False

    segments = list(self._seg_mgr._segments.keys())
    if segments:
      self._min_seconds = min(segments) * 60
      self._max_seconds = (max(segments) + 1) * 60
    return True

  def start(self, seconds: int = 0) -> None:
    self.seek_to(self._min_seconds + seconds, relative=False)

  def pause(self, pause: bool) -> None:
    if self._user_paused != pause:
      def update():
        print(f"{'paused...' if pause else 'resuming'} at {self.current_seconds:.2f} s")
        self._user_paused = pause
        return not pause
      self._interrupt_stream(update)

  def seek_to_flag(self, flag: FindFlag) -> None:
    next_time = self._timeline.find(self.current_seconds, flag)
    if next_time is not None:
      self.seek_to(next_time - 2, relative=False)  # seek 2 seconds before

  def seek_to(self, seconds: float, relative: bool) -> None:
    target_time = seconds + self.current_seconds if relative else seconds
    target_time = max(0.0, target_time)
    target_segment = int(target_time / 60)

    if not self._seg_mgr.has_segment(target_segment):
      print(f"Invalid seek to {target_time:.2f} s (segment {target_segment})")
      return

    print(f"Seeking to {int(target_time)} s, segment {target_segment}")
    if self.on_seeking:
      self.on_seeking(target_time)

    def update():
      self._current_segment = target_segment
      self._cur_mono_time = self._route_start_ts + int(target_time * 1e9)
      self._seeking_to = target_time
      return False

    self._interrupt_stream(update)
    self._seg_mgr.set_current_segment(target_segment)
    self._check_seek_progress()

  def _interrupt_stream(self, update_fn: Callable[[], bool]) -> None:
    self._interrupt.set()
    with self._stream_cv:
      self._events_ready = update_fn()
      if self._user_paused:
        self._interrupt.set()
      else:
        self._interrupt.clear()
      self._stream_cv.notify_all()

  def _check_seek_progress(self) -> None:
    event_data = self._seg_mgr.get_event_data()
    if not event_data.is_segment_loaded(self._current_segment):
      return

    seek_to = self._seeking_to
    self._seeking_to = -1.0
    if seek_to >= 0 and self.on_seeked_to:
      self.on_seeked_to(seek_to)

    # Resume stream
    self._interrupt_stream(lambda: True)

  def _handle_segment_merge(self) -> None:
    if self._exit:
      return

    event_data = self._seg_mgr.get_event_data()
    if self._stream_thread is None and event_data.segments:
      first_seg = min(event_data.segments.keys())
      self._start_stream(event_data.segments[first_seg])

    if self.on_segments_merged:
      self.on_segments_merged()

    self._interrupt_stream(lambda: False)
    self._check_seek_progress()

  def _start_stream(self, segment) -> None:
    events = segment.events
    if not events:
      return

    self._route_start_ts = events[0].logMonoTime
    self._cur_mono_time = self._route_start_ts - 1

    # Get datetime from initData
    for evt in events:
      if evt.which() == 'initData':
        wall_time = evt.initData.wallTimeNanos
        if wall_time > 0:
          self._route_date_time = wall_time // 1000000
        break

    # Write CarParams
    for evt in events:
      if evt.which() == 'carParams':
        self._car_fingerprint = evt.carParams.carFingerprint
        try:
          params = Params()
          car_params_bytes = evt.carParams.as_builder().to_bytes()
          params.put("CarParams", car_params_bytes)
          params.put("CarParamsPersistent", car_params_bytes)
        except Exception as e:
          print(f"failed to write CarParams: {e}")
        break

    # Start camera server
    if not self.has_flag(ReplayFlags.NO_VIPC):
      camera_sizes = {}
      for cam_name, cam_type in [('road', CameraType.ROAD), ('driver', CameraType.DRIVER), ('wide', CameraType.WIDE_ROAD)]:
        if cam_name in segment.frame_readers:
          fr = segment.frame_readers[cam_name]
          camera_sizes[cam_type] = (fr.w, fr.h)
      if camera_sizes:
        self._camera_server = CameraServer(camera_sizes)

    # Initialize timeline
    self._timeline.initialize(
      self._seg_mgr.route,
      self._route_start_ts,
      not (self._flags & ReplayFlags.NO_FILE_CACHE),
      lambda lr: self.on_qlog_loaded(lr) if self.on_qlog_loaded else None
    )

    self._stream_thread = threading.Thread(target=self._stream_thread_fn, daemon=True)
    self._stream_thread.start()

  def _stream_thread_fn(self) -> None:
    with self._stream_lock:
      while True:
        # Wait for events to be ready
        self._stream_cv.wait_for(lambda: self._exit or (self._events_ready and not self._interrupt.is_set()))
        if self._exit:
          break

        event_data = self._seg_mgr.get_event_data()
        events = event_data.events

        # Find first event after current time
        first_idx = 0
        for i, evt in enumerate(events):
          if evt.logMonoTime > self._cur_mono_time:
            first_idx = i
            break
        else:
          print("waiting for events...")
          self._events_ready = False
          continue

        last_idx = self._publish_events(events, first_idx)

        # Wait for camera frames to be sent
        if self._camera_server:
          self._camera_server.wait_for_sent()

        # Handle loop
        if last_idx >= len(events) and not self.has_flag(ReplayFlags.NO_LOOP):
          segments = list(self._seg_mgr._segments.keys())
          if segments and event_data.is_segment_loaded(max(segments)):
            print("reaches the end of route, restart from beginning")
            self._stream_lock.release()
            self.seek_to(self._min_seconds, relative=False)
            self._stream_lock.acquire()

  def _publish_events(self, events: list, first_idx: int) -> int:
    evt_start_ts = self._cur_mono_time
    loop_start_ts = time.monotonic_ns()
    prev_speed = self._speed

    idx = first_idx
    while idx < len(events) and not self._interrupt.is_set():
      evt = events[idx]

      # Update current segment
      segment = int((evt.logMonoTime - self._route_start_ts) / 1e9 / 60)
      if self._current_segment != segment:
        self._current_segment = segment
        self._seg_mgr.set_current_segment(segment)

      self._cur_mono_time = evt.logMonoTime

      # Check if service is enabled
      which = evt.which()
      if not self._sockets.get(which, False):
        idx += 1
        continue

      # Timing
      current_nanos = time.monotonic_ns()
      time_diff = (evt.logMonoTime - evt_start_ts) / self._speed - (current_nanos - loop_start_ts)

      # Reset timing if needed
      if time_diff < -1e9 or time_diff >= 1e9 or self._speed != prev_speed:
        evt_start_ts = evt.logMonoTime
        loop_start_ts = current_nanos
        prev_speed = self._speed
      elif time_diff > 0:
        # Interruptible sleep
        wait_secs = time_diff / 1e9
        if self._interrupt.wait(timeout=wait_secs):
          break  # Interrupted

      if self._interrupt.is_set():
        break

      # Publish message or frame
      if which in ('roadEncodeIdx', 'driverEncodeIdx', 'wideRoadEncodeIdx'):
        if self._camera_server:
          self._publish_frame(evt, which)
      else:
        self._publish_message(evt)

      idx += 1

    return idx

  def _publish_message(self, evt) -> None:
    if self._event_filter and self._event_filter(evt):
      return

    which = evt.which()
    if not self._sm:
      try:
        msg_bytes = evt.as_builder().to_bytes()
        self._pm.send(which, msg_bytes)
      except Exception as e:
        print(f"stop publishing {which} due to error: {e}")
        self._sockets[which] = False

  def _publish_frame(self, evt, which: str) -> None:
    cam_type = {
      'roadEncodeIdx': CameraType.ROAD,
      'driverEncodeIdx': CameraType.DRIVER,
      'wideRoadEncodeIdx': CameraType.WIDE_ROAD,
    }.get(which)

    if cam_type is None:
      return

    # Check if camera is enabled
    if cam_type == CameraType.DRIVER and not self.has_flag(ReplayFlags.DCAM):
      return
    if cam_type == CameraType.WIDE_ROAD and not self.has_flag(ReplayFlags.ECAM):
      return

    # Get frame reader for this segment
    event_data = self._seg_mgr.get_event_data()
    eidx = getattr(evt, which)
    seg_num = eidx.segmentId

    if seg_num in event_data.segments:
      seg_data = event_data.segments[seg_num]
      cam_name = {CameraType.ROAD: 'road', CameraType.DRIVER: 'driver', CameraType.WIDE_ROAD: 'wide'}[cam_type]
      if cam_name in seg_data.frame_readers:
        fr = seg_data.frame_readers[cam_name]
        if self._speed > 1.0:
          self._camera_server.wait_for_sent()
        self._camera_server.push_frame(cam_type, fr, evt)
