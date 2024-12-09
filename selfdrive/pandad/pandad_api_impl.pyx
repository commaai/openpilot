# distutils: language = c++
# cython: language_level=3
from cython.operator cimport dereference as deref, preincrement as preinc
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from libc.stdint cimport uint8_t, uint32_t, uint64_t

cdef extern from "panda.h":
  cdef struct can_frame:
    long address
    string dat
    long src

cdef extern from "opendbc/can/common.h":
  cdef struct CanFrame:
    long src
    uint32_t address
    vector[uint8_t] dat

  cdef struct CanData:
    uint64_t nanos
    vector[CanFrame] frames

cdef extern from "can_list_to_can_capnp.cc":
  void can_list_to_can_capnp_cpp(const vector[can_frame] &can_list, string &out, bool sendcan, bool valid)
  void can_capnp_to_can_list_cpp(const vector[string] &strings, vector[CanData] &can_data, bool sendcan)

def can_list_to_can_capnp(can_msgs, msgtype='can', valid=True):
  cdef can_frame *f
  cdef vector[can_frame] can_list

  can_list.reserve(len(can_msgs))
  for can_msg in can_msgs:
    f = &(can_list.emplace_back())
    f.address = can_msg[0]
    f.dat = can_msg[1]
    f.src = can_msg[2]

  cdef string out
  can_list_to_can_capnp_cpp(can_list, out, msgtype == 'sendcan', valid)
  return out

def can_capnp_to_list(strings, msgtype='can'):
  cdef vector[CanData] data
  can_capnp_to_can_list_cpp(strings, data, msgtype == 'sendcan')

  result = []
  cdef CanData *d
  cdef vector[CanData].iterator it = data.begin()
  while it != data.end():
    d = &deref(it)
    frames = [(f.address, (<char *>&f.dat[0])[:f.dat.size()], f.src) for f in d.frames]
    result.append((d.nanos, frames))
    preinc(it)
  return result
