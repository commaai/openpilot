# distutils: language = c++
#cython: language_level=3

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.set cimport set
from libc.stdint cimport uint32_t, uint64_t
from libcpp cimport bool, int

cdef extern from "cereal/visionipc/visionbuf.h":
  struct _cl_mem
  ctypedef _cl_mem * cl_mem

  cdef enum VisionStreamType:
    pass

  cdef cppclass VisionBuf:
    void * addr
    bool rgb
    size_t len
    size_t width
    size_t height
    size_t stride
    size_t uv_offset
    cl_mem buf_cl
    void set_frame_id(uint64_t id)

cdef extern from "cereal/visionipc/visionipc.h":
  struct VisionIpcBufExtra:
    uint32_t frame_id
    uint64_t timestamp_sof
    uint64_t timestamp_eof
    bool valid

cdef extern from "cereal/visionipc/visionipc_server.h":
  string get_endpoint_name(string, VisionStreamType)

  cdef cppclass VisionIpcServer:
    VisionIpcServer(string, void*, void*)
    void create_buffers(VisionStreamType, size_t, bool, size_t, size_t)
    void create_buffers_with_sizes(VisionStreamType, size_t, bool, size_t, size_t, size_t, size_t, size_t)
    VisionBuf * get_buffer(VisionStreamType)
    void send(VisionBuf *, VisionIpcBufExtra *, bool)
    void start_listener()

cdef extern from "cereal/visionipc/visionipc_client.h":
  cdef cppclass VisionIpcClient:
    int num_buffers
    VisionBuf buffers[1]
    VisionIpcClient(string, VisionStreamType, bool, void*, void*)
    VisionBuf * recv(VisionIpcBufExtra *, int)
    bool connect(bool)
    bool is_connected()
    @staticmethod
    set[VisionStreamType] getAvailableStreams(string, bool)
