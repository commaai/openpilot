# distutils: language = c++

from libcpp.string cimport string

from msgq.visionipc.visionipc cimport cl_context

cdef extern from "sunnypilot/modeld/runners/thneedmodel.h":
  cdef cppclass ThneedModel:
    ThneedModel(string, float*, size_t, int, bool, cl_context)
