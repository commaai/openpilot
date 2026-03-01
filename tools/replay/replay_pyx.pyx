# distutils: language = c++
# cython: language_level = 3

from libc.stdint cimport uint32_t, uint64_t, int64_t
from libc.time cimport time_t
from libcpp cimport bool
from libcpp.map cimport map as cppmap
from libcpp.string cimport string
from libcpp.vector cimport vector

# -- C++ declarations --

cdef extern from "tools/replay/util.h":
  cpdef enum class ReplyMsgType:
    Info
    Debug
    Warning
    Critical

  ctypedef void (*ReplayMessageHandler_raw)(ReplyMsgType, const string)
  void installMessageHandler "installMessageHandler" (ReplayMessageHandler_raw) except +

cdef extern from "tools/replay/py_downloader.h":
  ctypedef void (*DownloadProgressHandler_raw)(uint64_t, uint64_t, bool)
  void installDownloadProgressHandler "installDownloadProgressHandler" (DownloadProgressHandler_raw) except +

  ctypedef void (*DownloadHandler_raw)(const string&, bool, string*)
  void installDownloadHandler "installDownloadHandler" (DownloadHandler_raw) except +

cdef extern from "tools/replay/route.h":
  cpdef enum REPLAY_FLAGS:
    REPLAY_FLAG_NONE         = 0x0000
    REPLAY_FLAG_DCAM         = 0x0002
    REPLAY_FLAG_ECAM         = 0x0004
    REPLAY_FLAG_NO_LOOP      = 0x0010
    REPLAY_FLAG_NO_FILE_CACHE = 0x0020
    REPLAY_FLAG_QCAMERA      = 0x0040
    REPLAY_FLAG_NO_HW_DECODER = 0x0100
    REPLAY_FLAG_NO_VIPC      = 0x0400
    REPLAY_FLAG_ALL_SERVICES = 0x0800
    REPLAY_FLAG_BENCHMARK    = 0x1000

  cdef struct SegmentFile:
    pass

  cdef cppclass Route:
    const string& name()
    const cppmap[int, SegmentFile]& segments()

cdef extern from "tools/replay/replay.h":
  cdef extern string DEMO_ROUTE

  cdef cppclass Replay:
    Replay(vector[string], vector[string], void*, uint32_t) except +
    void addSegment(int, const string&, const string&, const string&, const string&, const string&, const string&) except +
    void setRouteName(const string&) except +
    void setRouteDateTime(time_t) except +
    bool load() except + nogil
    void start(int) except + nogil
    void pause(bool) except + nogil
    void seekTo(double, bool) except + nogil
    bool isPaused()
    void setSegmentCacheLimit(int)
    int segmentCacheLimit()
    void setSpeed(float)
    float getSpeed()
    double currentSeconds()
    double minSeconds()
    double maxSeconds()
    time_t routeDateTime()
    uint64_t routeStartNanos()
    const string& carFingerprint()
    const Route& route()
    void setLoop(bool)
    bool loop()
    void waitForFinished() except + nogil

# -- Module-level callback storage --

cdef object _py_message_handler = None
cdef object _py_progress_handler = None
cdef object _py_download_handler = None

cdef void _message_trampoline(ReplyMsgType msg_type, const string msg) noexcept with gil:
  global _py_message_handler
  handler = _py_message_handler
  if handler is not None:
    try:
      handler(msg_type, msg.decode("utf-8", errors="replace"))
    except BaseException:
      pass

cdef void _progress_trampoline(uint64_t cur, uint64_t total, bool success) noexcept with gil:
  global _py_progress_handler
  handler = _py_progress_handler
  if handler is not None:
    try:
      handler(cur, total, success)
    except BaseException:
      pass

cdef void _download_trampoline(const string& url, bool use_cache, string* out) noexcept with gil:
  global _py_download_handler
  handler = _py_download_handler
  if handler is not None:
    try:
      result = handler(url.decode("utf-8"), use_cache)
      if result:
        out[0] = result.encode("utf-8")
    except BaseException:
      pass

# Re-export TimelineType and FindFlag from Python timeline module
from openpilot.tools.replay.timeline_py import TimelineType, FindFlag

def get_demo_route():
  return DEMO_ROUTE.decode("utf-8")

# -- PyReplay wrapper --

cdef class PyReplay:
  cdef Replay* _r
  cdef object _qlog_paths      # dict[int, str]
  cdef object _timeline         # Timeline instance
  cdef object _timeline_started # bool
  cdef object _wait_stop        # threading.Event for cancelling the wait thread
  cdef object _wait_thread      # threading.Thread

  def __cinit__(self, str route, list allow=None, list block=None,
                uint32_t flags=REPLAY_FLAG_NONE, str data_dir=""):
    cdef vector[string] c_allow
    cdef vector[string] c_block
    if allow:
      for s in allow:
        c_allow.push_back(s.encode())
    if block:
      for s in block:
        c_block.push_back(s.encode())
    self._r = new Replay(c_allow, c_block, NULL, flags)

    self._timeline = None
    self._timeline_started = False
    self._qlog_paths = {}
    self._wait_stop = None
    self._wait_thread = None

    # Install download handler
    _ensure_cache_dir()
    global _py_download_handler
    _py_download_handler = _do_download
    installDownloadHandler(_download_trampoline)

    # Discover route in Python and populate C++ segments
    self._discover_route(route, data_dir if data_dir else None)

  def _discover_route(self, str route_str, data_dir):
    """Use tools/lib/route.py to discover segments and populate C++."""
    import re
    import time as time_mod
    from openpilot.tools.lib.route import Route as LibRoute

    # Parse route string: extract route name and optional segment range.
    # Supports formats:
    #   dongle_id|timestamp          (full route)
    #   dongle_id|timestamp--N       (segments N through end, old C++ compat)
    #   dongle_id|timestamp/N        (single segment N)
    #   dongle_id|timestamp/N:M      (segments N through M inclusive)
    #   dongle_id|timestamp/:M       (segments 0 through M inclusive)
    #   dongle_id|timestamp/N:       (segments N through end)
    begin_segment = 0
    end_segment = -1  # -1 means "to the end"

    # Match: optional_dongle_id SEPARATOR timestamp OPTIONAL(-- or / RANGE)
    m = re.match(r'^((?:[a-z0-9]{16}[|_/])?.{20})(?:(?:--|/)(.+))?$', route_str)
    if m:
      route_name = m.group(1).replace('_', '|').replace('/', '|')
      range_str = m.group(2)
      if range_str:
        # Determine separator between route and range using match positions
        sep = route_str[m.end(1):m.start(2)]
        if ':' in range_str:
          parts = range_str.split(':')
          begin_segment = int(parts[0]) if parts[0] else 0
          end_segment = int(parts[1]) if len(parts) > 1 and parts[1] else -1
        elif sep == '--':
          # route--N means "segments N through end" (old C++ compat)
          begin_segment = int(range_str)
          end_segment = -1
        else:
          # route/N means "only segment N"
          begin_segment = int(range_str)
          end_segment = int(range_str)
    else:
      route_name = route_str

    # Discover route
    py_route = LibRoute(route_name, data_dir=data_dir)
    canonical = str(py_route.name.canonical_name)

    # Parse datetime from timestamp (old format only; new format yields 0)
    dt = 0
    try:
      t = time_mod.strptime(py_route.name.time_str, "%Y-%m-%d--%H-%M-%S")
      dt = int(time_mod.mktime(t))
    except (ValueError, AttributeError):
      pass

    self._r.setRouteName(canonical.encode("utf-8"))
    self._r.setRouteDateTime(<time_t>dt)

    # Build segment map — single pass over segments
    qlog_map = {}
    for seg in py_route.segments:
      n = seg.name.segment_num

      # Apply segment range filter (inclusive on both ends, matching old C++ behavior)
      if n < begin_segment:
        continue
      if end_segment >= 0 and n > end_segment:
        continue

      rlog = seg.log_path or ""
      qlog = seg.qlog_path or ""
      if not rlog and not qlog:
        continue

      self._r.addSegment(n,
                         rlog.encode("utf-8"),
                         qlog.encode("utf-8"),
                         (seg.camera_path or "").encode("utf-8"),
                         (seg.dcamera_path or "").encode("utf-8"),
                         (seg.ecamera_path or "").encode("utf-8"),
                         (seg.qcamera_path or "").encode("utf-8"))

      if qlog:
        qlog_map[n] = qlog

    self._qlog_paths = qlog_map

  def __dealloc__(self):
    global _py_message_handler, _py_progress_handler, _py_download_handler
    # Stop wait thread if running
    if self._wait_stop is not None:
      self._wait_stop.set()
    if self._wait_thread is not None:
      self._wait_thread.join(timeout=2)
    # Stop timeline thread
    if self._timeline is not None:
      self._timeline.stop()
      self._timeline = None
    # Make trampolines no-ops while GIL is held
    _py_message_handler = None
    _py_progress_handler = None
    _py_download_handler = None
    # Clear C++ handlers (quick, GIL-safe)
    installMessageHandler(NULL)
    installDownloadProgressHandler(NULL)
    installDownloadHandler(NULL)
    if self._r != NULL:
      # Release GIL so C++ background threads can finish
      with nogil:
        del self._r
      self._r = NULL

  def load(self) -> bool:
    cdef bool ok
    with nogil:
      ok = self._r.load()
    return ok

  def start(self, int seconds=0):
    with nogil:
      self._r.start(seconds)
    # Start timeline building after streaming begins
    self._maybe_start_timeline()

  def _maybe_start_timeline(self):
    if self._timeline_started:
      return
    self._timeline_started = True

    import threading
    self._wait_stop = threading.Event()
    stop_event = self._wait_stop
    qlog_paths = self._qlog_paths

    def _wait_and_build():
      # Wait for route_start_nanos to be set
      import time as time_mod
      for _ in range(300):  # up to 30 seconds
        if stop_event.is_set():
          return
        ts = self._r.routeStartNanos()
        if ts != 0:
          break
        time_mod.sleep(0.1)
      else:
        return

      from openpilot.tools.replay.timeline_py import Timeline
      tl = Timeline()
      self._timeline = tl
      tl.build_async(qlog_paths, ts)

    self._wait_thread = threading.Thread(target=_wait_and_build, daemon=True)
    self._wait_thread.start()

  def pause(self, bool pause):
    with nogil:
      self._r.pause(pause)

  def is_paused(self) -> bool:
    return self._r.isPaused()

  def seek_to(self, double seconds, bool relative):
    with nogil:
      self._r.seekTo(seconds, relative)

  def seek_to_flag(self, flag):
    """Seek to next timeline event matching flag (pure Python)."""
    if self._timeline is None:
      return
    cur = self.current_seconds()
    ts = self._timeline.find(cur, flag)
    if ts is not None:
      self.seek_to(ts - 2, False)

  def set_speed(self, float speed):
    self._r.setSpeed(speed)

  def get_speed(self) -> float:
    return self._r.getSpeed()

  def current_seconds(self) -> float:
    return self._r.currentSeconds()

  def min_seconds(self) -> float:
    return self._r.minSeconds()

  def max_seconds(self) -> float:
    return self._r.maxSeconds()

  def route_date_time(self) -> int:
    return <int64_t>self._r.routeDateTime()

  def car_fingerprint(self) -> str:
    return self._r.carFingerprint().decode("utf-8")

  def route_name(self) -> str:
    return self._r.route().name().decode("utf-8")

  def segment_count(self) -> int:
    return self._r.route().segments().size()

  def set_segment_cache_limit(self, int n):
    self._r.setSegmentCacheLimit(n)

  def segment_cache_limit(self) -> int:
    return self._r.segmentCacheLimit()

  def set_loop(self, bool loop):
    self._r.setLoop(loop)

  def loop(self) -> bool:
    return self._r.loop()

  def wait_for_finished(self):
    with nogil:
      self._r.waitForFinished()

  def get_timeline(self) -> list:
    """Return timeline entries as list of (start_time, end_time, type_int) tuples."""
    if self._timeline is not None:
      return self._timeline.get_entries()
    return []

  def install_message_handler(self, handler):
    """Install a Python callable(msg_type: ReplyMsgType, msg: str) as the message handler."""
    global _py_message_handler
    _py_message_handler = handler
    if handler is not None:
      installMessageHandler(_message_trampoline)
    else:
      installMessageHandler(NULL)

  def install_download_progress_handler(self, handler):
    """Install a Python callable(cur: int, total: int, success: bool) as the download progress handler."""
    global _py_progress_handler
    _py_progress_handler = handler
    if handler is not None:
      installDownloadProgressHandler(_progress_trampoline)
    else:
      installDownloadProgressHandler(NULL)


# -- Download handler implementation --

cdef bint _cache_dir_ensured = False

def _ensure_cache_dir():
  """Create the download cache directory once."""
  global _cache_dir_ensured
  if not _cache_dir_ensured:
    import os
    from openpilot.system.hardware.hw import Paths
    os.makedirs(Paths.download_cache_root(), exist_ok=True)
    _cache_dir_ensured = True

def _do_download(url, use_cache):
  """Download a URL to local cache, return local file path."""
  import os
  import shutil
  import tempfile
  from openpilot.tools.lib.file_downloader import cache_file_path
  from openpilot.tools.lib.url_file import URLFile

  if not url.startswith("http://") and not url.startswith("https://"):
    return url

  # Always use cache path (even when use_cache=False, avoids temp file leak)
  local_path = cache_file_path(url)
  if use_cache and os.path.exists(local_path):
    return local_path

  try:
    uf = URLFile(url, cache=False)
    total = uf.get_length()
    if total <= 0:
      return ""

    from openpilot.system.hardware.hw import Paths
    tmp_fd, tmp_path = tempfile.mkstemp(dir=Paths.download_cache_root())
    try:
      downloaded = 0
      chunk_size = 1024 * 1024
      with os.fdopen(tmp_fd, 'wb') as f:
        while downloaded < total:
          data = uf.read(min(chunk_size, total - downloaded))
          if not data:
            break
          f.write(data)
          downloaded += len(data)
          handler = _py_progress_handler
          if handler is not None:
            try:
              handler(downloaded, total, True)
            except Exception:
              pass

      shutil.move(tmp_path, local_path)
      return local_path
    except Exception:
      try:
        os.unlink(tmp_path)
      except OSError:
        pass
      raise
  except Exception:
    return ""
