# distutils: language = c++
# cython: language_level=3
from cython.operator cimport dereference as deref, preincrement as preinc
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from libc.stdint cimport uint8_t, uint32_t, uint64_t

cdef extern from "selfdrive/pandad/can_types.h":
  cdef struct CanFrame:
    long src
    uint32_t address
    vector[uint8_t] dat

  cdef struct CanData:
    uint64_t nanos
    vector[CanFrame] frames

cdef extern from "can_list_to_can_capnp.cc":
  void can_list_to_can_capnp_cpp(const vector[CanFrame] &can_list, string &out, bool sendcan, bool valid) nogil
  void can_capnp_to_can_list_cpp(const vector[string] &strings, vector[CanData] &can_data, bool sendcan)

def can_list_to_can_capnp(can_msgs, msgtype='can', valid=True):
  cdef CanFrame *f
  cdef vector[CanFrame] can_list
  cdef uint32_t cpp_can_msgs_len = len(can_msgs)

  with nogil:
    can_list.reserve(cpp_can_msgs_len)

  for can_msg in can_msgs:
    f = &(can_list.emplace_back())
    f.address = can_msg[0]
    f.dat = can_msg[1]
    f.src = can_msg[2]

  cdef string out
  cdef bool is_sendcan = (msgtype == 'sendcan')
  cdef bool is_valid = valid
  with nogil:
    can_list_to_can_capnp_cpp(can_list, out, is_sendcan, is_valid)
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
