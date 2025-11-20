# distutils: language = c++
# cython: c_string_encoding=ascii, language_level=3

import numpy as np
cimport numpy as cnp
from libc.string cimport memcpy
from libc.stdint cimport uintptr_t

from msgq.visionipc.visionipc cimport cl_mem
from msgq.visionipc.visionipc_pyx cimport VisionBuf, CLContext as BaseCLContext
from .commonmodel cimport CL_DEVICE_TYPE_DEFAULT, cl_get_device_id, cl_create_context, cl_release_context
from .commonmodel cimport mat3, ModelFrame as cppModelFrame, DrivingModelFrame as cppDrivingModelFrame, MonitoringModelFrame as cppMonitoringModelFrame


cdef class CLContext(BaseCLContext):
  def __cinit__(self):
    self.device_id = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT)
    self.context = cl_create_context(self.device_id)

  def __dealloc__(self):
    if self.context:
      cl_release_context(self.context)

cdef class CLMem:
  @staticmethod
  cdef create(void * cmem):
    mem = CLMem()
    mem.mem = <cl_mem*> cmem
    return mem

  @property
  def mem_address(self):
    return <uintptr_t>(self.mem)

cdef class ModelFrame:
  cdef cppModelFrame * frame
  cdef int buf_size

  def __cinit__(self, CLContext context):
    self.frame = new cppModelFrame(context.device_id, context.context)

  def __dealloc__(self):
    del self.frame

  def array_from_vision_buf(self, VisionBuf vbuf):
    cdef unsigned char * data3
    data3 = self.frame.array_from_vision_buf(&vbuf.buf.buf_cl)
    return np.asarray(<cnp.uint8_t[:(vbuf.width*vbuf.height*3//2)]> data3)

  def cl_from_vision_buf(self, VisionBuf vbuf):
    cdef cl_mem * data4
    data4 = self.frame.cl_from_vision_buf(&vbuf.buf.buf_cl)
    return  CLMem.create(data4)

cdef class DrivingModelFrame(ModelFrame):
  cdef cppDrivingModelFrame * _frame

  def __cinit__(self, CLContext context, int temporal_skip):
    self._frame = new cppDrivingModelFrame(context.device_id, context.context, temporal_skip)
    self.frame = <cppModelFrame*>(self._frame)
    self.buf_size = self._frame.buf_size

cdef class MonitoringModelFrame(ModelFrame):
  cdef cppMonitoringModelFrame * _frame

  def __cinit__(self, CLContext context):
    self._frame = new cppMonitoringModelFrame(context.device_id, context.context)
    self.frame = <cppModelFrame*>(self._frame)
    self.buf_size = self._frame.buf_size

