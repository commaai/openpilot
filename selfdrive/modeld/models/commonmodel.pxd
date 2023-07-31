# distutils: language = c++
#cython: language_level=3

from libcpp cimport int, float
from .cl_pyx cimport _cl_device_id, _cl_context, _cl_mem

cdef extern from "common/mat.h":
  cdef struct mat3:
    float v[9]

cdef extern from "selfdrive/modeld/models/commonmodel.h":
  cppclass ModelFrame:
    int buf_size
    ModelFrame(_cl_device_id*, _cl_context*)
    float * prepare(_cl_mem*, int, int, int, int, mat3, _cl_mem**)
