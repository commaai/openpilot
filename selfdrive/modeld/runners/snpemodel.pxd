# distutils: language = c++
#cython: language_level=3

from libcpp.string cimport string
from cereal.visionipc.visionipc cimport cl_context

cdef extern from "selfdrive/modeld/runners/snpemodel.h":
  cdef cppclass SNPEModel:
    SNPEModel(string, float*, size_t, int, bool, cl_context)
