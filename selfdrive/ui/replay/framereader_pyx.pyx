# distutils: language = c++
# cython: language_level = 3
from libcpp cimport bool
from libcpp.string cimport string
from framereader_pxd cimport FrameReader as cpp_FrameReader

import os
from common.basedir import BASEDIR

cdef class FrameReader:
  cdef cpp_FrameReader* fr

  def __cinit__(self):
    self.fr = new cpp_FrameReader()

  def __dealloc__(self):
    del self.fr

  def load(self, file):
    return self.fr.load(file)
