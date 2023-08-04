# distutils: language = c++
#cython: language_level=3

from libcpp.string cimport string

cdef extern from "selfdrive/modeld/runners/runmodel.h":
  cdef cppclass RunModel:
    void addInput(string, float*, int)
    void setInputBuffer(string, float*, int)
    void * getCLBuffer(string)
    void execute()
