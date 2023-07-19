# distutils: language = c++
#cython: language_level=3

from libcpp.string cimport string
from libcpp cimport bool, int, float
from selfdrive.modeld.models.cl_pyx cimport _cl_context

cdef extern from "selfdrive/modeld/runners/runmodel.h":
  cdef int USE_CPU_RUNTIME
  cdef int USE_GPU_RUNTIME
  cdef int USE_DSP_RUNTIME

cdef extern from "selfdrive/modeld/runners/onnxmodel.h":
  cdef cppclass ONNXModel:
    ONNXModel(string, float*, size_t, int, bool, _cl_context*)
    void addInput(string, float*, int)
    void setInputBuffer(string, float*, int)
    void * getCLBuffer(string)
    void execute()
