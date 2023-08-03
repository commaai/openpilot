# distutils: language = c++
# cython: c_string_encoding=ascii, language_level=3

from cereal.visionipc.visionipc cimport cl_device_id, cl_context, cl_mem, CL_DEVICE_TYPE_DEFAULT
from cereal.visionipc.visionipc_pyx cimport CLContext as BaseCLContext
from .cl_pyx cimport cl_get_device_id, cl_create_context

cdef class CLContext(BaseCLContext):
  def __cinit__(self):
    self.device_id = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT)
    self.context = cl_create_context(self.device_id)

cdef class CLMem:
  @staticmethod
  cdef create(void * cmem):
    mem = CLMem()
    mem.mem = <cl_mem*> cmem
    return mem
