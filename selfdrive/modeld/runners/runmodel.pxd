# distutils: language = c++

from libcpp.string cimport string

cdef extern from "selfdrive/modeld/runners/runmodel.h":
  cdef int USE_CPU_RUNTIME
  cdef int USE_GPU_RUNTIME
  cdef int USE_DSP_RUNTIME

  cdef cppclass RunModel:
    void addInput(string, float*, int)
    void setInputBuffer(string, float*, int)
    void * getCLBuffer(string)
    void execute()
