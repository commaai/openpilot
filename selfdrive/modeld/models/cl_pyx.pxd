# distutils: language = c++
#cython: language_level=3

from libcpp cimport int, float

cdef extern from "<CL/cl.h>":
  cdef unsigned long CL_DEVICE_TYPE_DEFAULT

  struct _cl_device_id
  struct _cl_context
  struct _cl_mem

cdef extern from "common/clutil.h":
  _cl_device_id * cl_get_device_id(unsigned long)
  _cl_context * cl_create_context(_cl_device_id*)

cdef class CLContext:
  cdef _cl_device_id * device_id
  cdef _cl_context * context

cdef class CLMem:
  cdef _cl_mem ** mem;
