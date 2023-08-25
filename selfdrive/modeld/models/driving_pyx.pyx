# distutils: language = c++
# cython: c_string_encoding=ascii

import numpy as np
cimport numpy as cnp
from libcpp cimport bool
from libc.string cimport memcpy
from libc.stdint cimport uint32_t, uint64_t

from .commonmodel cimport mat3
from .driving cimport FEATURE_LEN as CPP_FEATURE_LEN, HISTORY_BUFFER_LEN as CPP_HISTORY_BUFFER_LEN, DESIRE_LEN as CPP_DESIRE_LEN, \
                      TRAFFIC_CONVENTION_LEN as CPP_TRAFFIC_CONVENTION_LEN, DRIVING_STYLE_LEN as CPP_DRIVING_STYLE_LEN, \
                      NAV_FEATURE_LEN as CPP_NAV_FEATURE_LEN, NAV_INSTRUCTION_LEN as CPP_NAV_INSTRUCTION_LEN, \
                      OUTPUT_SIZE as CPP_OUTPUT_SIZE, NET_OUTPUT_SIZE as CPP_NET_OUTPUT_SIZE, MODEL_FREQ as CPP_MODEL_FREQ, CPP_USE_THNEED
from .driving cimport MessageBuilder, PublishState as cppPublishState
from .driving cimport fill_model_msg, fill_pose_msg, update_calibration as cpp_update_calibration

FEATURE_LEN = CPP_FEATURE_LEN
HISTORY_BUFFER_LEN = CPP_HISTORY_BUFFER_LEN
DESIRE_LEN = CPP_DESIRE_LEN
TRAFFIC_CONVENTION_LEN = CPP_TRAFFIC_CONVENTION_LEN
DRIVING_STYLE_LEN = CPP_DRIVING_STYLE_LEN
NAV_FEATURE_LEN = CPP_NAV_FEATURE_LEN
NAV_INSTRUCTION_LEN = CPP_NAV_INSTRUCTION_LEN
OUTPUT_SIZE = CPP_OUTPUT_SIZE
NET_OUTPUT_SIZE = CPP_NET_OUTPUT_SIZE
MODEL_FREQ = CPP_MODEL_FREQ
USE_THNEED = CPP_USE_THNEED

cdef class PublishState:
  cdef cppPublishState state

def update_calibration(float[:] device_from_calib_euler, bool wide_camera, bool bigmodel_frame):
  cdef mat3 result = cpp_update_calibration(&device_from_calib_euler[0], wide_camera, bigmodel_frame)
  np_result = np.empty(9, dtype=np.float32)
  cdef float[:] np_result_view = np_result
  memcpy(&np_result_view[0], &result.v[0], 9*sizeof(float))
  return np_result.reshape(3, 3)

def create_model_msg(float[:] model_outputs, PublishState ps, uint32_t vipc_frame_id, uint32_t vipc_frame_id_extra, uint32_t frame_id, float frame_drop,
                     uint64_t timestamp_eof, uint64_t timestamp_llk, float model_execution_time, bool nav_enabled, bool valid):
  cdef MessageBuilder msg
  fill_model_msg(msg, &model_outputs[0], ps.state, vipc_frame_id, vipc_frame_id_extra, frame_id, frame_drop,
                 timestamp_eof, timestamp_llk, model_execution_time, nav_enabled, valid)

  output_size = msg.getSerializedSize()
  output_data = bytearray(output_size)
  cdef unsigned char * output_ptr = output_data
  assert msg.serializeToBuffer(output_ptr, output_size) > 0, "output buffer is too small to serialize"
  return bytes(output_data)

def create_pose_msg(float[:] model_outputs, uint32_t vipc_frame_id, uint32_t vipc_dropped_frames, uint64_t timestamp_eof, bool valid):
  cdef MessageBuilder msg
  fill_pose_msg(msg, &model_outputs[0], vipc_frame_id, vipc_dropped_frames, timestamp_eof, valid)

  output_size = msg.getSerializedSize()
  output_data = bytearray(output_size)
  cdef unsigned char * output_ptr = output_data
  assert msg.serializeToBuffer(output_ptr, output_size) > 0, "output buffer is too small to serialize"
  return bytes(output_data)
