# distutils: language = c++
# cython: language_level = 3

from libc.stdint cimport uint32_t, uint64_t, int64_t
from libc.time cimport time_t
from libcpp cimport bool
from libcpp.map cimport map as cppmap
from libcpp.memory cimport shared_ptr
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

cdef extern from "tools/replay/timeline.h":
    cdef enum class CTimelineType "TimelineType":
        _None "TimelineType::None"
        Engaged "TimelineType::Engaged"
        AlertInfo "TimelineType::AlertInfo"
        AlertWarning "TimelineType::AlertWarning"
        AlertCritical "TimelineType::AlertCritical"
        UserBookmark "TimelineType::UserBookmark"

    cpdef enum class FindFlag:
        nextEngagement
        nextDisEngagement
        nextUserBookmark
        nextInfo
        nextWarning
        nextCritical

    cdef cppclass _TimelineEntry "Timeline::Entry":
        double start_time
        double end_time
        CTimelineType type

cdef extern from "tools/replay/route.h":
    cdef struct SegmentFile:
        pass
    cdef cppclass Route:
        const string& name()
        const cppmap[int, SegmentFile]& segments()

cdef extern from "tools/replay/replay.h":
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

    cdef extern string DEMO_ROUTE

    cdef cppclass Replay:
        Replay(const string&, vector[string], vector[string], void*, uint32_t, const string&, bool) except +
        bool load() except + nogil
        void start(int) except + nogil
        void pause(bool) except + nogil
        void seekTo(double, bool) except + nogil
        void seekToFlag(FindFlag) except + nogil
        bool isPaused()
        void setSegmentCacheLimit(int)
        int segmentCacheLimit()
        void setSpeed(float)
        float getSpeed()
        double currentSeconds()
        double minSeconds()
        double maxSeconds()
        time_t routeDateTime()
        const string& carFingerprint()
        const Route& route()
        shared_ptr[vector[_TimelineEntry]] getTimeline()
        void setLoop(bool)
        bool loop()

# -- Module-level callback storage --

cdef object _py_message_handler = None
cdef object _py_progress_handler = None

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

# Python-accessible TimelineType constants (cdef enum class can't be cpdef due to None keyword)
cdef int _TL_ENGAGED = int(CTimelineType.Engaged)
cdef int _TL_ALERT_INFO = int(CTimelineType.AlertInfo)
cdef int _TL_ALERT_WARNING = int(CTimelineType.AlertWarning)
cdef int _TL_ALERT_CRITICAL = int(CTimelineType.AlertCritical)
cdef int _TL_USER_BOOKMARK = int(CTimelineType.UserBookmark)

class TimelineType:
    Engaged = _TL_ENGAGED
    AlertInfo = _TL_ALERT_INFO
    AlertWarning = _TL_ALERT_WARNING
    AlertCritical = _TL_ALERT_CRITICAL
    UserBookmark = _TL_USER_BOOKMARK

def get_demo_route():
    return DEMO_ROUTE.decode("utf-8")

def formatted_data_size(size_t size):
    return formattedDataSize(size).decode("utf-8")

# -- PyReplay wrapper --

cdef class PyReplay:
    cdef Replay* _r

    def __cinit__(self, str route, list allow=None, list block=None,
                  uint32_t flags=REPLAY_FLAG_NONE, str data_dir="",
                  bool auto_source=False):
        cdef string c_route = route.encode()
        cdef vector[string] c_allow
        cdef vector[string] c_block
        cdef string c_data_dir = data_dir.encode()
        if allow:
            for s in allow:
                c_allow.push_back(s.encode())
        if block:
            for s in block:
                c_block.push_back(s.encode())
        self._r = new Replay(c_route, c_allow, c_block, NULL, flags, c_data_dir, auto_source)

    def __dealloc__(self):
        global _py_message_handler, _py_progress_handler
        _py_message_handler = None
        _py_progress_handler = None
        installMessageHandler(NULL)
        installDownloadProgressHandler(NULL)
        if self._r != NULL:
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

    def pause(self, bool pause):
        with nogil:
            self._r.pause(pause)

    def is_paused(self) -> bool:
        return self._r.isPaused()

    def seek_to(self, double seconds, bool relative):
        with nogil:
            self._r.seekTo(seconds, relative)

    def seek_to_flag(self, FindFlag flag):
        with nogil:
            self._r.seekToFlag(flag)

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
        cdef shared_ptr[vector[_TimelineEntry]] entries = self._r.getTimeline()
        cdef vector[_TimelineEntry]* vec = entries.get()
        cdef list result = []
        cdef size_t i
        cdef size_t n = vec.size()
        for i in range(n):
            result.append((vec.at(i).start_time, vec.at(i).end_time, int(vec.at(i).type)))
        return result

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
