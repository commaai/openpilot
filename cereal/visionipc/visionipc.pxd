# distutils: language = c++
#cython: language_level=3

from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stdint cimport uint32_t, uint64_t
from libcpp cimport bool

cdef extern from "visionbuf.h":
  cdef enum VisionStreamType:
    pass

  cdef cppclass VisionBuf:
    void * addr
    size_t len

cdef extern from "visionipc.h":
  struct VisionIpcBufExtra:
    uint32_t frame_id
    uint64_t timestamp_sof
    uint64_t timestamp_eof

cdef extern from "visionipc_server.h":
  cdef cppclass VisionIpcServer:
    VisionIpcServer(string, void*, void*)
    void create_buffers(VisionStreamType, size_t, bool, size_t, size_t)
    VisionBuf * get_buffer(VisionStreamType)
    void send(VisionBuf *, VisionIpcBufExtra *, bool)
    void start_listener()
