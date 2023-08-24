# distutils: language = c++

from libcpp.string cimport string

from cereal.visionipc.visionipc cimport cl_context

cdef extern from "selfdrive/modeld/runners/thneedmodel.h":
  cdef cppclass ThneedModel:
    ThneedModel(string, float*, size_t, int, bool, cl_context)
