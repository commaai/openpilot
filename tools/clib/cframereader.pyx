# distutils: language = c++
# cython: language_level=3

cdef extern from "FrameReader.hpp":
  cdef cppclass CFrameReader "FrameReader":
    CFrameReader(const char *)
    char *get(int)

cdef class FrameReader():
  cdef CFrameReader *fr

  def __cinit__(self, fn):
    self.fr = new CFrameReader(fn)

  def __dealloc__(self):
    del self.fr

  def get(self, idx):
    self.fr.get(idx)

