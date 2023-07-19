# distutils: language = c++
# cython: c_string_encoding=ascii, language_level=3

from .cl_pyx cimport CLContext, CLMem
from .commonmodel cimport ModelFrame as cppModelFrame

cdef class ModelFrame:
  cdef cppModelFrame * frame

  def __cinit__(self, CLContext context):
    self.frame = new cppModelFrame(context.device_id, context.context)

  def __dealloc__(self):
    del self.frame

  # def prepare(self, CLMem yuv_cl, int frame_width, int frame_height, int frame_stride, int frame_uv_offset, float[:] projection, CLMem output):
  #   return self.frame.prepare(*yuv_cl.mem, frame_width, frame_height, frame_stride, frame_uv_offset, projection, output.mem)
