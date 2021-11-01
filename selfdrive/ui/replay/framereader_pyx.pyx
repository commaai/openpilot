# distutils: language = c++
# cython: language_level = 3
from libcpp cimport bool
from libcpp.string cimport string
import numpy as np
cimport numpy as cnp
from cython.view cimport array
from libc.string cimport memcpy

cdef extern from "selfdrive/ui/replay/framereader.h":
  cdef cppclass cpp_FrameReader "FrameReader":
    cpp_FrameReader()
    bool load(string)
    int getRGBSize()
    int getYUVSize()
    int frame_count()
    void *get(int)
    void *get_yuv(int)
    int width
    int height

cdef class FrameReader:
  cdef cpp_FrameReader* fr

  def __cinit__(self):
    self.fr = new cpp_FrameReader()

  def __dealloc__(self):
    del self.fr

  def load(self, file):
    return self.fr.load(file.encode())

  @property
  def width(self):
    return self.fr.width

  @property
  def height(self):
    return self.fr.height

  @property
  def rgbSize(self):
    return self.fr.getRGBSize()
  
  @property
  def yuvSize(self):
    return self.fr.getYUVSize()
  
  @property
  def frame_count(self):
    return self.fr.frame_count()

  def get(self, id):
    addr = self.fr.get(id)
    if not addr:
      return None
    cdef cnp.ndarray dat = np.empty(self.rgbSize, dtype=np.uint8)
    cdef char[:] dat_view = dat
    memcpy(&dat_view[0], addr, self.rgbSize)
    return dat

  def get_yuv(self, id):
    addr = self.fr.get_yuv(id)
    if (not addr):
      return None
    cdef cnp.ndarray dat = np.empty(self.yuvSize, dtype=np.uint8)
    cdef char[:] dat_view = dat
    memcpy(&dat_view[0], addr, self.yuvSize)
    return dat
