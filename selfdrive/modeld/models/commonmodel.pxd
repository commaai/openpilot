# distutils: language = c++
#cython: language_level=3

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport int, float

cdef extern from "selfdrive/modeld/models/commonmodel.h":
  cppclass ModelFrame:
    ModelFrame(cl_device_id device_id, cl_context context)
    float * prepare(cl_mem yuv_cl, int width, int height, int frame_stride, int frame_uv_offset, mat3 transform, cl_mem * output)
