# distutils: language = c++
# cython: language_level=3
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

cdef struct can_frame:
  long address
  string dat
  long busTime
  long src

cdef extern void can_list_to_can_capnp_cpp(const vector[can_frame] &can_list, string &out, bool sendCan, bool valid)

def can_list_to_can_capnp(can_msgs, msgtype='can', valid=True):
  cdef vector[can_frame] can_list
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
