# distutils: language = c++

from libcpp.string cimport string

from msgq.visionipc.visionipc cimport cl_context

cdef extern from "selfdrive/modeld/runners/snpemodel.h":
  cdef cppclass SNPEModel:
    SNPEModel(string, float*, size_t, int, bool, cl_context)
