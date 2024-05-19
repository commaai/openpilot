# distutils: language = c++
# cython: language_level=3
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "panda.h":
  cdef struct can_frame:
    long address
    string dat
    long busTime
    long src

cdef extern from "can_list_to_can_capnp.cc":
  void can_list_to_can_capnp_cpp(const vector[can_frame] &can_list, string &out, bool sendCan, bool valid)

def can_list_to_can_capnp(can_msgs, msgtype='can', valid=True):
  cdef can_frame *f
  cdef vector[can_frame] can_list

  can_list.reserve(len(can_msgs))
  for can_msg in can_msgs:
    f = &(can_list.emplace_back())
    f.address = can_msg[0]
    f.busTime = can_msg[1]
    f.dat = can_msg[2]
    f.src = can_msg[3]

  cdef string out
  can_list_to_can_capnp_cpp(can_list, out, msgtype == 'sendcan', valid)
  return out
