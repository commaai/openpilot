# distutils: language = c++
# cython: c_string_encoding=ascii, language_level=3

import sys
import numpy as np
cimport numpy as cnp
from cython.view cimport array
from libc.string cimport memcpy
from libc.stdint cimport uint32_t, uint64_t
from libcpp cimport bool
from libcpp.string cimport string

from .visionipc cimport VisionIpcServer as cppVisionIpcServer
from .visionipc cimport VisionIpcClient as cppVisionIpcClient
from .visionipc cimport VisionBuf as cppVisionBuf
from .visionipc cimport VisionIpcBufExtra

cpdef enum VisionStreamType:
  VISION_STREAM_ROAD
  VISION_STREAM_DRIVER
  VISION_STREAM_WIDE_ROAD
  VISION_STREAM_MAP


cdef class VisionIpcServer:
  cdef cppVisionIpcServer * server

  def __init__(self, string name):
    self.server = new cppVisionIpcServer(name, NULL, NULL)

  def create_buffers(self, VisionStreamType tp, size_t num_buffers, bool rgb, size_t width, size_t height):
    self.server.create_buffers(tp, num_buffers, rgb, width, height)

  def create_buffers_with_sizes(self, VisionStreamType tp, size_t num_buffers, bool rgb, size_t width, size_t height, size_t size, size_t stride, size_t uv_offset):
    self.server.create_buffers_with_sizes(tp, num_buffers, rgb, width, height, size, stride, uv_offset)

  def send(self, VisionStreamType tp, const unsigned char[:] data, uint32_t frame_id=0, uint64_t timestamp_sof=0, uint64_t timestamp_eof=0):
    cdef cppVisionBuf * buf = self.server.get_buffer(tp)

    # Populate buffer
    assert buf.len == len(data)
    memcpy(buf.addr, &data[0], len(data))
    buf.set_frame_id(frame_id)

    cdef VisionIpcBufExtra extra
    extra.frame_id = frame_id
    extra.timestamp_sof = timestamp_sof
    extra.timestamp_eof = timestamp_eof

    self.server.send(buf, &extra, False)

  def start_listener(self):
    self.server.start_listener()

  def __dealloc__(self):
    del self.server


cdef class VisionIpcClient:
  cdef cppVisionBuf * buf
  cdef cppVisionIpcClient * client

  def __cinit__(self, string name, VisionStreamType stream, bool conflate):
    self.client = new cppVisionIpcClient(name, stream, conflate, NULL, NULL)
    self.buf = NULL

  def __dealloc__(self):
    del self.client

  @property
  def width(self):
    return None if not self.buf else self.buf.width

  @property
  def height(self):
    return None if not self.buf else self.buf.height

  @property
  def stride(self):
    return None if not self.buf else self.buf.stride

  @property
  def uv_offset(self):
    return None if not self.buf else self.buf.uv_offset

  def recv(self, int timeout_ms=100):
    self.buf = self.client.recv(NULL, timeout_ms)
    if not self.buf:
      return None
    cdef cnp.ndarray dat = np.empty(self.buf.len, dtype=np.uint8)
    cdef char[:] dat_view = dat
    memcpy(&dat_view[0], self.buf.addr, self.buf.len)
    return dat

  def connect(self, bool blocking):
    return self.client.connect(blocking)

  def is_connected(self):
    return self.client.is_connected()
