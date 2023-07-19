# distutils: language = c++
#cython: language_level=3

from libcpp.string cimport string
from libcpp cimport bool, int, float

cdef extern from "<CL/cl.h>":
  cdef int CL_DEVICE_TYPE_DEFAULT

  struct _cl_device_id:
    pass
  struct _cl_context:
    pass

cdef extern from "selfdrive/modeld/runners/onnxmodel.h":
  cdef cppclass ONNXModel:
    ONNXModel(string, float*, size_t, int, bool, _cl_context*)
    void addInput(string, float*, int)
    void setInputBuffer(string, float*, int)
    void * getCLBuffer(string)
    void execute()
