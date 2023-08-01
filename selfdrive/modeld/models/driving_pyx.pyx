# distutils: language = c++
# cython: c_string_encoding=ascii, language_level=3

import numpy as np
cimport numpy as cnp
from libcpp cimport bool, float
from libc.stdint cimport uint32_t, uint64_t

cdef extern from "selfdrive/modeld/models/driving.h":
  float * create_model_msg(float *, uint32_t, uint32_t, uint32_t, float, uint64_t, uint64_t, float, bool, bool);

def get_model_msg(float[:] model_outputs, uint32_t vipc_frame_id, uint32_t vipc_frame_id_extra, uint32_t frame_id, float frame_drop,
                  uint64_t timestamp_eof, uint64_t timestamp_llk, float model_execution_time, bool nav_enabled, bool valid):
  create_model_msg(&model_outputs[0], vipc_frame_id, vipc_frame_id_extra, frame_id, frame_drop,
                   timestamp_eof, timestamp_llk, model_execution_time, nav_enabled, valid)
