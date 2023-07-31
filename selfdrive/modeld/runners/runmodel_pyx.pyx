# distutils: language = c++
# cython: c_string_encoding=ascii, language_level=3

import numpy as np
cimport numpy as cnp
from cython.view cimport array
from libc.string cimport memcpy
from libcpp.string cimport string
from libcpp cimport bool, int, float

from .runmodel cimport USE_CPU_RUNTIME, USE_GPU_RUNTIME, USE_DSP_RUNTIME
from .runmodel cimport ONNXModel as cppONNXModel
from selfdrive.modeld.models.cl_pyx cimport CLContext, CLMem

class Runtime:
  CPU = USE_CPU_RUNTIME
  GPU = USE_GPU_RUNTIME
  DSP = USE_DSP_RUNTIME

cdef class ONNXModel:
  cdef cppONNXModel * model

  def __cinit__(self, string path, float[:] output, int runtime, bool use_tf8, CLContext context):
    self.model = new cppONNXModel(path, &output[0], len(output), runtime, use_tf8, context.context)

  def __dealloc__(self):
    del self.model

  def addInput(self, string name, float[:] buffer):
    if buffer is not None:
      self.model.addInput(name, &buffer[0], len(buffer))
    else:
      self.model.addInput(name, NULL, 0)

  def setInputBuffer(self, string name, float[:] buffer):
    self.model.setInputBuffer(name, &buffer[0], len(buffer))

  def getCLBuffer(self, string name):
    cdef void * cl_buf = self.model.getCLBuffer(name)
    if not cl_buf:
      return None
    return CLMem.create(cl_buf)

  def execute(self):
    self.model.execute()
