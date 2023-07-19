# distutils: language = c++
# cython: c_string_encoding=ascii, language_level=3

from .cl_pyx cimport _cl_device_id, _cl_context, _cl_mem, cl_get_device_id, cl_create_context, CL_DEVICE_TYPE_DEFAULT

cdef class CLContext:
  def __cinit__(self):
    self.device_id = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT)
    self.context = cl_create_context(self.device_id)

cdef class CLMem:
  def __cinit__(self):
    self.mem = NULL
