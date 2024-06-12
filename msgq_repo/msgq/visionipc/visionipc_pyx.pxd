# distutils: language = c++
#cython: language_level=3

from .visionipc cimport VisionBuf as cppVisionBuf
from .visionipc cimport cl_device_id, cl_context

cdef class CLContext:
  cdef cl_device_id device_id
  cdef cl_context context

cdef class VisionBuf:
  cdef cppVisionBuf * buf

  @staticmethod
  cdef create(cppVisionBuf*)
