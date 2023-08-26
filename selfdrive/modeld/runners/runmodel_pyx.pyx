# distutils: language = c++
# cython: c_string_encoding=ascii

from libcpp.string cimport string
from libc.string cimport memcpy

from selfdrive.modeld.models.commonmodel_pyx cimport CLMem

cdef class RunModel:
  def __dealloc__(self):
    del self.model

  def addInput(self, string name, float[:] buffer):
    if buffer is not None:
      self.model.addInput(name, &buffer[0], len(buffer))
    else:
      self.model.addInput(name, NULL, 0)

  def setInputBuffer(self, string name, float[:] buffer):
    if buffer is not None:
      self.model.setInputBuffer(name, &buffer[0], len(buffer))
    else:
      self.model.setInputBuffer(name, NULL, 0)

  def getCLBuffer(self, string name):
    cdef void * cl_buf = self.model.getCLBuffer(name)
    if not cl_buf:
      return None
    return CLMem.create(cl_buf)

  def execute(self):
    self.model.execute()
