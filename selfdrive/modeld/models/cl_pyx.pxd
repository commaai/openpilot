# distutils: language = c++
#cython: language_level=3

from libcpp cimport int, float, long
from cereal.visionipc.visionipc cimport cl_device_id, cl_context, cl_mem
from cereal.visionipc.visionipc_pyx cimport CLContext as BaseCLContext

cdef extern from "common/clutil.h":
  cl_device_id cl_get_device_id(unsigned long)
  cl_context cl_create_context(cl_device_id)

cdef class CLContext(BaseCLContext):
  pass

cdef class CLMem:
  cdef cl_mem * mem;

  @staticmethod
  cdef create(void*)
