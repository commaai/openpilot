# distutils: language = c++
# cython: language_level=3
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from libc.stdint cimport uint8_t, uint64_t

cdef extern from "panda.h":
  cdef struct can_frame:
    long address
    string dat
    long busTime
    long src

cdef extern from "opendbc/can/common.h":
  cdef struct CanFrame:
    long src
    long address
    vector[uint8_t] dat

  cdef struct CanData:
    uint64_t nanos
    vector[CanFrame] frames

cdef extern from "can_list_to_can_capnp.cc":
  void can_list_to_can_capnp_cpp(const vector[can_frame] &can_list, string &out, bool sendCan, bool valid)
  void can_capnp_to_can_list_cpp(const vector[string] &strings, vector[CanData] &can_data, bool sendcan)

def can_list_to_can_capnp(can_msgs, msgtype='can', valid=True):
  cdef vector[can_frame] can_list
  can_list.reserve(len(can_msgs))

  cdef can_frame f
  for can_msg in can_msgs:
    f.address = can_msg[0]
    f.busTime = can_msg[1]
    f.dat = can_msg[2]
    f.src = can_msg[3]
    can_list.push_back(f)
  cdef string out
  can_list_to_can_capnp_cpp(can_list, out, msgtype == 'sendcan', valid)
  return out

def can_capp_to_can_list(strings, sendcan):
  cdef vector[CanData] data
  can_capnp_to_can_list_cpp(strings, data, sendcan)
  result = []
  for c in data:
    frames = []
    for f in c.frames:
      frames.append([f.address, 0, f.dat, f.src])
    result.append([c.nanos, frames])
  return result
