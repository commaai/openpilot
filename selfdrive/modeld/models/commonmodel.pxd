# distutils: language = c++
#cython: language_level=3

from libcpp cimport int, float
from cereal.visionipc.visionipc cimport cl_device_id, cl_context, cl_mem

cdef extern from "common/mat.h":
  cdef struct mat3:
    float v[9]

cdef extern from "selfdrive/modeld/models/commonmodel.h":
  cppclass ModelFrame:
    int buf_size
    ModelFrame(cl_device_id, cl_context)
    float * prepare(cl_mem, int, int, int, int, mat3, cl_mem*)
