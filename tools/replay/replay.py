#!/usr/bin/env python3
import argparse
import logging
import resource
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

import cereal.messaging as messaging
from cereal.services import SERVICE_LIST
from openpilot.common.params import Params

from openpilot.tools.replay.camera import CameraServer, CameraType
from openpilot.tools.replay.seg_mgr import SegmentManager, ReplayFlags
from openpilot.tools.replay.timeline import Timeline, FindFlag

log = logging.getLogger("replay")


@dataclass
class ReplayStats:
  """Tracks timing statistics for playback observability."""
  time_buffer_ns: int = 0  # Current time_diff (positive = ahead of schedule)
  _lag_events: deque = field(default_factory=deque)  # (timestamp, lag_ns)
  _lag_threshold_ns: int = -10_000_000  # -10ms
  _window_secs: float = 30.0

  def record_timing(self, time_diff_ns: int) -> None:
    """Record a timing measurement. Called from _publish_events."""
    self.time_buffer_ns = time_diff_ns
    now = time.monotonic()

    # Record if it's a lag (behind schedule by more than threshold)
    if time_diff_ns < self._lag_threshold_ns:
      self._lag_events.append((now, time_diff_ns))

    # Prune entries older than window
    cutoff = now - self._window_secs
    while self._lag_events and self._lag_events[0][0] < cutoff:
      self._lag_events.popleft()

  @property
  def lag_count(self) -> int:
    """Number of lag events in the rolling window."""
    return len(self._lag_events)

  @property
  def worst_lag_ns(self) -> int:
    """Most negative time_diff in the rolling window (0 if none)."""
    if not self._lag_events:
      return 0
    return min(lag for _, lag in self._lag_events)


@dataclass
class BenchmarkStats:
  """Tracks benchmark timeline events."""
  process_start_ts: int = 0
  timeline: list = field(default_factory=list)  # [(timestamp_ns, description)]

  def record(self, description: str) -> None:
    self.timeline.append((time.monotonic_ns(), description))

DEMO_ROUTE = "a2a0ccea32023010|2023-07-27--13-01-19"


class Replay:
  def __init__(self, route: str, allow: Optional[list[str]] = None, block: Optional[list[str]] = None,
               sm=None, flags: int = 0, data_dir: str = ""):
    self._sm = sm
    self._flags = flags
    self._seg_mgr = SegmentManager(route, flags, data_dir)

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
    self._min_seconds = 0.0
    self._max_seconds = 0.0
    self._speed = 1.0
    self._car_fingerprint = ""

    # Timing stats for observability
    self._stats = ReplayStats()

    # Benchmark mode state
    self._benchmark_stats = BenchmarkStats()
    self._benchmark_stats.process_start_ts = time.monotonic_ns()
    self._benchmark_done = False
    self._benchmark_cv = threading.Condition()

    # Callbacks
    self.on_segments_merged: Optional[Callable[[], None]] = None
    self.on_seeking: Optional[Callable[[float], None]] = None
    self.on_seeked_to: Optional[Callable[[float], None]] = None
    self.on_qlog_loaded: Optional[Callable] = None
    self._event_filter: Optional[Callable] = None

  def __del__(self):
    if hasattr(self, '_stream_thread') and self._stream_thread is not None and self._stream_thread.is_alive():
      log.info("shutdown: in progress...")
      self._interrupt_stream(lambda: setattr(self, '_exit', True) or False)
      self._stream_thread.join()
      log.info("shutdown: done")

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

    log.info(f"active services: {', '.join(active_services)}")
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
  def stats(self) -> ReplayStats:
    return self._stats

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

  @property
  def benchmark_stats(self) -> BenchmarkStats:
    return self._benchmark_stats

  def wait_for_finished(self) -> None:
    """Wait for benchmark mode to complete."""
    with self._benchmark_cv:
      self._benchmark_cv.wait_for(lambda: self._benchmark_done)

  def load(self) -> bool:
    log.info(f"loading route {self._seg_mgr._route_name}")
    if not self._seg_mgr.load():
      return False

    segments = list(self._seg_mgr._segments.keys())
    if segments:
      self._min_seconds = min(segments) * 60
      self._max_seconds = (max(segments) + 1) * 60

    if self.has_flag(ReplayFlags.BENCHMARK):
      self._benchmark_stats.record("route metadata loaded")
    return True

  def start(self, seconds: int = 0) -> None:
    self.seek_to(self._min_seconds + seconds, relative=False)

  def pause(self, pause: bool) -> None:
    if self._user_paused != pause:
      def update():
        log.info(f"{'paused...' if pause else 'resuming'} at {self.current_seconds:.2f} s")
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
      log.warning(f"Invalid seek to {target_time:.2f} s (segment {target_segment})")
      return

    log.info(f"Seeking to {int(target_time)} s, segment {target_segment}")
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
          log.warning(f"failed to write CarParams: {e}")
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
        # Warm the cache before playback to prevent initial stutter
        for cam_name in segment.frame_readers:
          self._camera_server.warm_cache(segment.frame_readers[cam_name])

    # Initialize timeline
    self._timeline.initialize(
      self._seg_mgr.route,
      self._route_start_ts,
      lambda lr: self.on_qlog_loaded(lr) if self.on_qlog_loaded else None
    )

    if self.has_flag(ReplayFlags.BENCHMARK):
      self._benchmark_stats.record("streaming started")

    self._stream_thread = threading.Thread(target=self._stream_thread_fn, daemon=True)
    self._stream_thread.start()

  def _stream_thread_fn(self) -> None:
    benchmark_mode = self.has_flag(ReplayFlags.BENCHMARK)
    benchmark_segment_start: Optional[float] = None
    benchmark_start_segment = self._current_segment

    while True:
      # Hold lock only while checking/updating shared state
      with self._stream_lock:
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
          log.info("waiting for events...")
          self._events_ready = False
          continue

      if benchmark_mode and benchmark_segment_start is None:
        benchmark_segment_start = time.monotonic()

      # Publish WITHOUT holding lock - allows UI to interrupt quickly
      prev_segment = self._current_segment
      last_idx = self._publish_events(events, first_idx)

      # Wait for camera frames to be sent
      if self._camera_server:
        self._camera_server.wait_for_sent()

      # Track segment completion for benchmark
      if benchmark_mode and self._current_segment != prev_segment:
        elapsed_ms = (time.monotonic() - benchmark_segment_start) * 1000
        realtime_ms = 60 * 1000  # 60 seconds per segment
        multiplier = realtime_ms / elapsed_ms if elapsed_ms > 0 else 0
        self._benchmark_stats.record(f"segment {prev_segment} done publishing ({elapsed_ms:.0f} ms, {multiplier:.0f}x realtime)")
        benchmark_segment_start = time.monotonic()

        # In benchmark mode, exit after first segment
        if prev_segment == benchmark_start_segment:
          self._benchmark_stats.record("benchmark done")
          with self._benchmark_cv:
            self._benchmark_done = True
            self._benchmark_cv.notify_all()
          break

      # Handle loop
      if last_idx >= len(events) and not self.has_flag(ReplayFlags.NO_LOOP):
        segments = list(self._seg_mgr._segments.keys())
        if segments and event_data.is_segment_loaded(max(segments)):
          log.info("reaches the end of route, restart from beginning")
          self.seek_to(self._min_seconds, relative=False)

  def _publish_events(self, events: list, first_idx: int) -> int:
    evt_start_ts = self._cur_mono_time
    loop_start_ts = time.monotonic_ns()
    prev_speed = self._speed
    benchmark_mode = self.has_flag(ReplayFlags.BENCHMARK)

    idx = first_idx
    while idx < len(events) and not self._interrupt.is_set():
      evt = events[idx]

      # Update current segment
      segment = int((evt.logMonoTime - self._route_start_ts) / 1e9 / 60)
      if self._current_segment != segment:
        self._current_segment = segment
        self._seg_mgr.set_current_segment(segment)
        # In benchmark mode, return after segment change to allow tracking
        if benchmark_mode:
          return idx

      self._cur_mono_time = evt.logMonoTime

      # Check if service is enabled
      which = evt.which()
      if not self._sockets.get(which, False):
        idx += 1
        continue

      # Skip timing in benchmark mode for maximum throughput
      if not benchmark_mode:
        # Timing
        current_nanos = time.monotonic_ns()
        time_diff = (evt.logMonoTime - evt_start_ts) / self._speed - (current_nanos - loop_start_ts)

        # Record timing stats for observability
        self._stats.record_timing(int(time_diff))

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
        log.warning(f"stop publishing {which} due to error: {e}")
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
    # Note: eidx.segmentId is the local frame index, not the segment number
    # The segment number comes from the current playback position
    event_data = self._seg_mgr.get_event_data()

    if self._current_segment not in event_data.segments:
      return

    seg_data = event_data.segments[self._current_segment]
    cam_name = {CameraType.ROAD: 'road', CameraType.DRIVER: 'driver', CameraType.WIDE_ROAD: 'wide'}[cam_type]
    if cam_name not in seg_data.frame_readers:
      return

    fr = seg_data.frame_readers[cam_name]
    if self._speed > 1.0:
      self._camera_server.wait_for_sent()
    self._camera_server.push_frame(cam_type, fr, evt)


def main():
  # Increase file descriptor limit on macOS
  if sys.platform == 'darwin':
    try:
      resource.setrlimit(resource.RLIMIT_NOFILE, (1024, 1024))
    except Exception:
      pass

  parser = argparse.ArgumentParser(description='openpilot replay tool')
  parser.add_argument('route', nargs='?', default='', help='Route to replay')
  parser.add_argument('-a', '--allow', type=str, default='', help='Whitelist of services (comma-separated)')
  parser.add_argument('-b', '--block', type=str, default='', help='Blacklist of services (comma-separated)')
  parser.add_argument('-c', '--buffer', type=int, default=-1, help='Number of segments to buffer in memory')
  parser.add_argument('-s', '--start', type=int, default=0, help='Start from <seconds>')
  parser.add_argument('-x', '--playback', type=float, default=-1, help='Playback speed')
  parser.add_argument('-d', '--data_dir', type=str, default='', help='Local directory with routes')
  parser.add_argument('-p', '--prefix', type=str, default='', help='OPENPILOT_PREFIX')
  parser.add_argument('--demo', action='store_true', help='Use demo route')
  parser.add_argument('--dcam', action='store_true', help='Load driver camera')
  parser.add_argument('--ecam', action='store_true', help='Load wide road camera')
  parser.add_argument('--no-loop', action='store_true', help='Stop at end of route')
  parser.add_argument('--qcam', action='store_true', help='Load qcamera')
  parser.add_argument('--no-vipc', action='store_true', help='Do not output video')
  parser.add_argument('--all', action='store_true', help='Output all messages')
  parser.add_argument('--benchmark', action='store_true', help='Run in benchmark mode (process all events then exit with stats)')
  parser.add_argument('--headless', action='store_true', help='Run without UI')

  args = parser.parse_args()

  # Determine route
  route = args.route
  if args.demo:
    route = DEMO_ROUTE
  if not route:
    print("No route provided. Use --help for usage information.")
    return 1

  # Parse flags
  flags = ReplayFlags.NONE
  if args.dcam:
    flags |= ReplayFlags.DCAM
  if args.ecam:
    flags |= ReplayFlags.ECAM
  if args.no_loop:
    flags |= ReplayFlags.NO_LOOP
  if args.qcam:
    flags |= ReplayFlags.QCAMERA
  if args.no_vipc:
    flags |= ReplayFlags.NO_VIPC
  if args.all:
    flags |= ReplayFlags.ALL_SERVICES
  if args.benchmark:
    flags |= ReplayFlags.BENCHMARK

  # Parse allow/block lists
  allow = [s.strip() for s in args.allow.split(',') if s.strip()]
  block = [s.strip() for s in args.block.split(',') if s.strip()]

  # Set prefix if provided
  if args.prefix:
    import os
    os.environ['OPENPILOT_PREFIX'] = args.prefix

  # Create replay instance
  replay = Replay(
    route=route,
    allow=allow,
    block=block,
    flags=flags,
    data_dir=args.data_dir
  )

  if args.buffer > 0:
    replay.segment_cache_limit = args.buffer

  if args.playback > 0:
    replay.speed = max(0.2, min(8.0, args.playback))

  if not replay.load():
    return 1

  if args.benchmark:
    replay.start(args.start)
    replay.wait_for_finished()

    stats = replay.benchmark_stats
    process_start = stats.process_start_ts

    print("\n===== REPLAY BENCHMARK RESULTS =====")
    print(f"Route: {replay.route.name}")
    print("\nTIMELINE:")
    print("  t=0 ms        process start")
    for ts, event in stats.timeline:
      ms = (ts - process_start) / 1e6
      padding = " " * max(1, 8 - len(str(int(ms))))
      print(f"  t={ms:.0f} ms{padding}{event}")

    return 0

  replay.start(args.start)

  if args.headless:
    try:
      while True:
        time.sleep(5)
        print(f"replay: {replay.current_seconds:.1f}s / {replay.max_seconds:.1f}s")
    except KeyboardInterrupt:
      pass
    return 0

  from openpilot.tools.replay.consoleui import ConsoleUI
  console_ui = ConsoleUI(replay)
  return console_ui.exec()


if __name__ == '__main__':
  sys.exit(main())
