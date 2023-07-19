# distutils: language = c++
# cython: c_string_encoding=ascii, language_level=3

import numpy as np
cimport numpy as cnp
from cython.view cimport array
from libc.string cimport memcpy
from libcpp.string cimport string
from libcpp cimport bool, int, float

from .runmodel cimport ONNXModel as cppONNXModel
from .runmodel cimport _cl_context, _cl_device_id


cdef class CLContext:
  cdef _cl_device_id * device_id
  cdef _cl_context * context

  def __cinit__(self):
    pass


cdef class ONNXModel:
  cdef cppONNXModel * model

  def __cinit__(self, string path, float[:] output, int runtime, bool use_tf8, CLContext context):
    self.model = new cppONNXModel(path, &output[0], len(output), runtime, use_tf8, context.context)

  def __dealloc__(self):
    del self.model

  def addInput(self, string name, float[:] buffer):
    self.model.addInput(name, &buffer[0], len(buffer))

  def setInputBuffer(self, string name, float[:] buffer):
    self.model.setInputBuffer(name, &buffer[0], len(buffer))

  def execute(self):
    self.model.execute()
