# distutils: language = c++

from cereal.visionipc.visionipc cimport cl_mem
from cereal.visionipc.visionipc_pyx cimport CLContext as BaseCLContext

cdef class CLContext(BaseCLContext):
  pass

cdef class CLMem:
  cdef cl_mem * mem

  @staticmethod
  cdef create(void*)
