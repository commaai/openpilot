# distutils: language = c++
#cython: language_level=3

from libcpp.string cimport string
from cereal.visionipc.visionipc cimport cl_context

cdef extern from "selfdrive/modeld/runners/onnxmodel.h":
  cdef cppclass ONNXModel:
    ONNXModel(string, float*, size_t, int, bool, cl_context)
