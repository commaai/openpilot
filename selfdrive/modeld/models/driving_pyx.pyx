# distutils: language = c++
# cython: c_string_encoding=ascii, language_level=3

from libcpp cimport bool
from libc.stdint cimport uint32_t, uint64_t
from .driving cimport MessageBuilder, PublishState as cppPublishState
from .driving cimport fill_model_msg, fill_pose_msg

cdef class PublishState:
  cdef cppPublishState state

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
