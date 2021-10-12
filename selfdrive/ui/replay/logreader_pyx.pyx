# distutils: language = c++
# cython: language_level = 3
from libcpp cimport bool
from libcpp.string cimport string
from logreader_pxd cimport LogReader as cpp_LogReader

import os
from common.basedir import BASEDIR

cdef class LogReader:
  cdef cpp_LogReader* lr

  def __cinit__(self):
    self.lr = new cpp_LogReader()

  def __dealloc__(self):
    del self.lr

  def load(self, logfile):
    return self.lr.load(logfile)
