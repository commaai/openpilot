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
from .visionipc cimport get_endpoint_name as cpp_get_endpoint_name


def get_endpoint_name(string name, VisionStreamType stream):
  return cpp_get_endpoint_name(name, stream).decode('utf-8')


cpdef enum VisionStreamType:
  VISION_STREAM_ROAD
  VISION_STREAM_DRIVER
  VISION_STREAM_WIDE_ROAD
  VISION_STREAM_MAP


cdef class VisionBuf:
  @staticmethod
  cdef create(cppVisionBuf * cbuf):
    buf = VisionBuf()
    buf.buf = cbuf
    return buf

  @property
  def data(self):
    return np.asarray(<cnp.uint8_t[:self.buf.len]> self.buf.addr)

  @property
  def width(self):
    return self.buf.width

  @property
  def height(self):
    return self.buf.height

  @property
  def stride(self):
    return self.buf.stride

  @property
  def uv_offset(self):
    return self.buf.uv_offset


cdef class VisionIpcServer:
  cdef cppVisionIpcServer * server

  def __init__(self, string name):
    self.server = new cppVisionIpcServer(name, NULL, NULL)

  def create_buffers(self, VisionStreamType tp, size_t num_buffers, size_t width, size_t height):
    self.server.create_buffers(tp, num_buffers, width, height)

  def create_buffers_with_sizes(self, VisionStreamType tp, size_t num_buffers, size_t width, size_t height, size_t size, size_t stride, size_t uv_offset):
    self.server.create_buffers_with_sizes(tp, num_buffers, width, height, size, stride, uv_offset)

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
  cdef cppVisionIpcClient * client
  cdef VisionIpcBufExtra extra

  def __cinit__(self, string name, VisionStreamType stream, bool conflate, CLContext context = None):
    if context:
      self.client = new cppVisionIpcClient(name, stream, conflate, context.device_id, context.context)
    else:
      self.client = new cppVisionIpcClient(name, stream, conflate, NULL, NULL)

  def __dealloc__(self):
    del self.client

  @property
  def width(self):
    return self.client.buffers[0].width if self.client.num_buffers else None

  @property
  def height(self):
    return self.client.buffers[0].height if self.client.num_buffers else None

  @property
  def stride(self):
    return self.client.buffers[0].stride if self.client.num_buffers else None

  @property
  def uv_offset(self):
    return self.client.buffers[0].uv_offset if self.client.num_buffers else None

  @property
  def buffer_len(self):
    return self.client.buffers[0].len if self.client.num_buffers else None

  @property
  def num_buffers(self):
    return self.client.num_buffers

  @property
  def frame_id(self):
    return self.extra.frame_id

  @property
  def timestamp_sof(self):
    return self.extra.timestamp_sof

  @property
  def timestamp_eof(self):
    return self.extra.timestamp_eof

  @property
  def valid(self):
    return self.extra.valid

  def recv(self, int timeout_ms=100):
    buf = self.client.recv(&self.extra, timeout_ms)
    if not buf:
      return None
    return VisionBuf.create(buf)

  def connect(self, bool blocking):
    return self.client.connect(blocking)

  def is_connected(self):
    return self.client.is_connected()

  @staticmethod
  def available_streams(string name, bool block):
    return cppVisionIpcClient.getAvailableStreams(name, block)
