# distutils: language = c++
#cython: language_level=3

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport int, float

cdef extern from "selfdrive/modeld/runners/onnxmodel.h":
  cdef cppclass OnnxModel:
    void addInput(string name, float * buffer, int size)
    void setInputBuffer(string name, float * buffer, int size)
    void * getCLBuffer(string name)
    void execute()
