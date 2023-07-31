# distutils: language = c++
# cython: c_string_encoding=ascii, language_level=3

import numpy as np
cimport numpy as cnp
from libcpp cimport float
from libc.string cimport memcpy
from cereal.visionipc.visionipc_pyx cimport VisionBuf
from .cl_pyx cimport CLContext, CLMem
from .commonmodel cimport mat3, ModelFrame as cppModelFrame

cdef class ModelFrame:
  cdef cppModelFrame * frame

  def __cinit__(self, CLContext context):
    self.frame = new cppModelFrame(context.device_id, context.context)

  def __dealloc__(self):
    del self.frame

  def prepare(self, VisionBuf buf, float[:] projection, CLMem output):
    cdef mat3 cprojection
    memcpy(cprojection.v, &projection[0], 9)
    cdef float * data = self.frame.prepare(buf.buf.buf_cl, buf.width, buf.height, buf.stride, buf.uv_offset, cprojection, output.mem)
    if not data:
      return None
    return np.asarray(<cnp.float32_t[:self.frame.buf_size]> data)
