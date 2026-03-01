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
  string formattedDataSize(size_t) except +

cdef extern from "tools/replay/py_downloader.h":
  ctypedef void (*DownloadProgressHandler_raw)(uint64_t, uint64_t, bool)
  void installDownloadProgressHandler "installDownloadProgressHandler" (DownloadProgressHandler_raw) except +

  ctypedef string (*DownloadHandler_raw)(const string&, bool)
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
    except:
      pass

cdef void _progress_trampoline(uint64_t cur, uint64_t total, bool success) noexcept with gil:
  global _py_progress_handler
  handler = _py_progress_handler
  if handler is not None:
    try:
      handler(cur, total, success)
    except:
      pass

cdef string _download_trampoline(const string& url, bool use_cache) noexcept with gil:
  global _py_download_handler
  handler = _py_download_handler
  if handler is not None:
    try:
      result = handler(url.decode("utf-8"), use_cache)
      if result:
        return result.encode("utf-8")
    except:
      pass
  return string()

# Re-export TimelineType and FindFlag from Python timeline module
from openpilot.tools.replay.timeline_py import TimelineType, FindFlag

def get_demo_route():
  return DEMO_ROUTE.decode("utf-8")

def formatted_data_size(size_t size):
  return formattedDataSize(size).decode("utf-8")

# -- PyReplay wrapper --

cdef class PyReplay:
  cdef Replay* _r
  cdef object _route_name   # str
  cdef object _qlog_paths   # dict[int, str]
  cdef object _timeline     # Timeline instance
  cdef object _timeline_started  # bool

  def __cinit__(self, str route, list allow=None, list block=None,
                uint32_t flags=REPLAY_FLAG_NONE, str data_dir="",
                bool auto_source=False):
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

    # Install download handler
    self._install_download_handler()

    # Discover route in Python and populate C++ segments
    self._discover_route(route, data_dir if data_dir else None)

  def _install_download_handler(self):
    """Install a Python download handler that uses file_downloader logic."""
    global _py_download_handler
    _py_download_handler = _do_download
    installDownloadHandler(_download_trampoline)

  def _discover_route(self, str route_str, data_dir):
    """Use tools/lib/route.py to discover segments and populate C++."""
    import re
    import time as time_mod
    from openpilot.tools.lib.route import Route as LibRoute

    # Parse segment range (e.g. "route_name/0:5" or "route_name--5")
    # Simple parsing: split on / or -- to get range
    begin_segment = 0
    end_segment = -1
    route_name = route_str

    # Try to parse route/segment range
    # Format: dongle_id|timestamp/begin:end or dongle_id|timestamp--seg
    m = re.match(r'^([a-z0-9]{16}[|_/].{20})(?:(?:--|/)(.+))?$', route_str)
    if m:
      route_name = m.group(1).replace('_', '|').replace('/', '|')
      range_str = m.group(2)
      if range_str:
        if '/' in route_str and ':' in range_str:
          # Slash-separated range: route/begin:end
          parts = range_str.split(':')
          begin_segment = int(parts[0]) if parts[0] else 0
          end_segment = int(parts[1]) if len(parts) > 1 and parts[1] else -1
        elif ':' in range_str:
          parts = range_str.split(':')
          begin_segment = int(parts[0]) if parts[0] else 0
          end_segment = int(parts[1]) if len(parts) > 1 and parts[1] else -1
        else:
          # Single segment number
          begin_segment = int(range_str)
          end_segment = begin_segment
    else:
      route_name = route_str

    # Discover route using tools/lib/route.py
    py_route = LibRoute(route_name, data_dir=data_dir)
    canonical = str(py_route.name.canonical_name)

    # Parse datetime from timestamp
    dt = 0
    try:
      ts = py_route.name.time_str
      t = time_mod.strptime(ts, "%Y-%m-%d--%H-%M-%S")
      dt = int(time_mod.mktime(t))
    except (ValueError, AttributeError):
      pass

    self._r.setRouteName(canonical.encode("utf-8"))
    self._r.setRouteDateTime(<time_t>dt)

    # Build segment map
    log_paths = py_route.log_paths()
    qlog_paths = py_route.qlog_paths()
    camera_paths = py_route.camera_paths()
    dcamera_paths = py_route.dcamera_paths()
    ecamera_paths = py_route.ecamera_paths()
    qcamera_paths = py_route.qcamera_paths()

    qlog_map = {}
    for i in range(len(qlog_paths)):
      rlog = log_paths[i] if i < len(log_paths) and log_paths[i] else ""
      qlog = qlog_paths[i] if qlog_paths[i] else ""
      cam = camera_paths[i] if i < len(camera_paths) and camera_paths[i] else ""
      dcam = dcamera_paths[i] if i < len(dcamera_paths) and dcamera_paths[i] else ""
      ecam = ecamera_paths[i] if i < len(ecamera_paths) and ecamera_paths[i] else ""
      qcam = qcamera_paths[i] if i < len(qcamera_paths) and qcamera_paths[i] else ""

      if not rlog and not qlog:
        continue

      # Apply segment range filter
      if begin_segment > 0 and i < begin_segment:
        continue
      if end_segment >= 0 and i > end_segment:
        continue

      self._r.addSegment(i,
                         rlog.encode("utf-8"),
                         qlog.encode("utf-8"),
                         cam.encode("utf-8"),
                         dcam.encode("utf-8"),
                         ecam.encode("utf-8"),
                         qcam.encode("utf-8"))

      if qlog:
        qlog_map[i] = qlog

    self._route_name = canonical
    self._qlog_paths = qlog_map

  def __dealloc__(self):
    global _py_message_handler, _py_progress_handler, _py_download_handler
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
    def _wait_and_build():
      # Wait for route_start_nanos to be set
      import time as time_mod
      for _ in range(300):  # up to 30 seconds
        ts = self._r.routeStartNanos()
        if ts != 0:
          break
        time_mod.sleep(0.1)
      else:
        return

      from openpilot.tools.replay.timeline_py import Timeline
      tl = Timeline()
      self._timeline = tl
      tl.build_async(self._qlog_paths, ts)

    t = threading.Thread(target=_wait_and_build, daemon=True)
    t.start()

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

def _do_download(url, use_cache):
  """Download a URL to local cache, return local file path."""
  import hashlib
  import os
  import shutil
  import tempfile
  from openpilot.system.hardware.hw import Paths
  from openpilot.tools.lib.url_file import URLFile

  if not url.startswith("http://") and not url.startswith("https://"):
    # Local file, return as-is
    return url

  if use_cache:
    url_without_query = url.split("?")[0]
    local_path = os.path.join(Paths.download_cache_root(), hashlib.sha256(url_without_query.encode()).hexdigest())
    if os.path.exists(local_path):
      return local_path
  else:
    local_path = None

  try:
    uf = URLFile(url, cache=False)
    total = uf.get_length()
    if total <= 0:
      return ""

    os.makedirs(Paths.download_cache_root(), exist_ok=True)
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
          # Report progress via the existing progress handler
          handler = _py_progress_handler
          if handler is not None:
            try:
              handler(downloaded, total, True)
            except:
              pass

      if local_path is not None:
        shutil.move(tmp_path, local_path)
        return local_path
      else:
        return tmp_path
    except Exception:
      try:
        os.unlink(tmp_path)
      except OSError:
        pass
      raise
  except Exception:
    return ""
