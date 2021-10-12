# distutils: language = c++
# cython: language_level = 3
from libcpp cimport bool
from libcpp.string cimport string
from logreader_pxd cimport LogReader as c_LogReader

import os
from common.basedir import BASEDIR

cdef class LogReader:
  cdef c_LogReader* lr

  def __cinit__(self):
    self.lr = new c_LogReader()

  def __dealloc__(self):
    del self.lr

  def load(self, logfile, is_bz2file):
    return self.lr.load(logfile, is_bz2file)
