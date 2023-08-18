# distutils: language = c++
#cython: language_level=3

from libcpp cimport bool
from libc.stdint cimport uint32_t, uint64_t
from .commonmodel cimport mat3

cdef extern from "cereal/messaging/messaging.h":
  cdef cppclass MessageBuilder:
    size_t getSerializedSize()
    int serializeToBuffer(unsigned char *, size_t)

cdef extern from "selfdrive/modeld/models/driving.h":
  cdef bool CPP_USE_THNEED
  cdef struct PublishState: pass

  mat3 update_calibration(float *, bool, bool)
  void fill_model_msg(MessageBuilder, float *, PublishState, uint32_t, uint32_t, uint32_t, float, uint64_t, uint64_t, float, bool, bool)
  void fill_pose_msg(MessageBuilder, float *, uint32_t, uint32_t, uint64_t, bool)
