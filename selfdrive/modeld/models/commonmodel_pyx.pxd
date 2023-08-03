# distutils: language = c++
#cython: language_level=3

from libcpp cimport int, float, long
from cereal.visionipc.visionipc cimport cl_device_id, cl_context, cl_mem
from cereal.visionipc.visionipc_pyx cimport CLContext as BaseCLContext

cdef class CLContext(BaseCLContext):
  pass

cdef class CLMem:
  cdef cl_mem * mem;

  @staticmethod
  cdef create(void*)
