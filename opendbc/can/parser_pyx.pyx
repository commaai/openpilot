# distutils: language = c++
# cython: c_string_encoding=ascii, language_level=3

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from libc.stdint cimport uint32_t, uint64_t, uint16_t
from libcpp cimport bool
from libcpp.map cimport map

from .common cimport CANParser as cpp_CANParser
from .common cimport SignalParseOptions, MessageParseOptions, dbc_lookup, SignalValue, DBC

import os
import numbers
from collections import defaultdict

cdef int CAN_INVALID_CNT = 5


cdef class CANParser:
  cdef:
    cpp_CANParser *can
    const DBC *dbc
    map[string, uint32_t] msg_name_to_address
    map[uint32_t, string] address_to_msg_name
    vector[SignalValue] can_values

  cdef readonly:
    dict vl
    dict vl_all
    bool can_valid
    string dbc_name
    int can_invalid_cnt

  def __init__(self, dbc_name, signals, checks=None, bus=0, enforce_checks=True):
    if checks is None:
      checks = []

    self.dbc_name = dbc_name
    self.dbc = dbc_lookup(dbc_name)
    if not self.dbc:
      raise RuntimeError(f"Can't find DBC: {dbc_name}")

    self.vl = {}
    self.vl_all = {}
    self.can_valid = False
    self.can_invalid_cnt = CAN_INVALID_CNT

    cdef int i
    cdef int num_msgs = self.dbc[0].num_msgs
    for i in range(num_msgs):
      msg = self.dbc[0].msgs[i]
      name = msg.name.decode('utf8')

      self.msg_name_to_address[name] = msg.address
      self.address_to_msg_name[msg.address] = name
      self.vl[msg.address] = {}
      self.vl[name] = self.vl[msg.address]
      self.vl_all[msg.address] = defaultdict(list)
      self.vl_all[name] = self.vl_all[msg.address]

    # Convert message names into addresses
    for i in range(len(signals)):
      s = signals[i]
      if not isinstance(s[1], numbers.Number):
        name = s[1].encode('utf8')
        s = (s[0], self.msg_name_to_address[name])
        signals[i] = s

    for i in range(len(checks)):
      c = checks[i]
      if not isinstance(c[0], numbers.Number):
        name = c[0].encode('utf8')
        c = (self.msg_name_to_address[name], c[1])
        checks[i] = c

    if enforce_checks:
      checked_addrs = {c[0] for c in checks}
      signal_addrs = {s[1] for s in signals}
      unchecked = signal_addrs - checked_addrs
      if len(unchecked):
        err_msg = ', '.join(f"{self.address_to_msg_name[addr].decode()} ({hex(addr)})" for addr in unchecked)
        raise RuntimeError(f"Unchecked addrs: {err_msg}")

    cdef vector[SignalParseOptions] signal_options_v
    cdef SignalParseOptions spo
    for sig_name, sig_address in signals:
      spo.address = sig_address
      spo.name = sig_name
      signal_options_v.push_back(spo)

    message_options = dict((address, 0) for _, address in signals)
    message_options.update(dict(checks))

    cdef vector[MessageParseOptions] message_options_v
    cdef MessageParseOptions mpo
    for msg_address, freq in message_options.items():
      mpo.address = msg_address
      mpo.check_frequency = freq
      message_options_v.push_back(mpo)

    self.can = new cpp_CANParser(bus, dbc_name, message_options_v, signal_options_v)
    self.update_vl()

  cdef unordered_set[uint32_t] update_vl(self):
    cdef unordered_set[uint32_t] updated_addrs

    # Update invalid flag
    self.can_invalid_cnt += 1
    if self.can.can_valid:
      self.can_invalid_cnt = 0
    self.can_valid = self.can_invalid_cnt < CAN_INVALID_CNT

    new_vals = self.can.query_latest()
    for cv in new_vals:
      # Cast char * directly to unicode
      cv_name = <unicode>cv.name
      self.vl[cv.address][cv_name] = cv.value
      self.vl_all[cv.address][cv_name].extend(cv.all_values)
      updated_addrs.insert(cv.address)

    return updated_addrs

  def update_string(self, dat, sendcan=False):
    for v in self.vl_all.values():
      v.clear()

    self.can.update_string(dat, sendcan)
    return self.update_vl()

  def update_strings(self, strings, sendcan=False):
    for v in self.vl_all.values():
      v.clear()

    updated_addrs = set()
    for s in strings:
      self.can.update_string(s, sendcan)
      updated_addrs.update(self.update_vl())
    return updated_addrs


cdef class CANDefine():
  cdef:
    const DBC *dbc

  cdef public:
    dict dv
    string dbc_name

  def __init__(self, dbc_name):
    self.dbc_name = dbc_name
    self.dbc = dbc_lookup(dbc_name)
    if not self.dbc:
      raise RuntimeError(f"Can't find DBC: '{dbc_name}'")

    num_vals = self.dbc[0].num_vals

    address_to_msg_name = {}

    num_msgs = self.dbc[0].num_msgs
    for i in range(num_msgs):
      msg = self.dbc[0].msgs[i]
      name = msg.name.decode('utf8')
      address = msg.address
      address_to_msg_name[address] = name

    dv = defaultdict(dict)

    for i in range(num_vals):
      val = self.dbc[0].vals[i]

      sgname = val.name.decode('utf8')
      def_val = val.def_val.decode('utf8')
      address = val.address
      msgname = address_to_msg_name[address]

      # separate definition/value pairs
      def_val = def_val.split()
      values = [int(v) for v in def_val[::2]]
      defs = def_val[1::2]

      # two ways to lookup: address or msg name
      dv[address][sgname] = dict(zip(values, defs))
      dv[msgname][sgname] = dv[address][sgname]

    self.dv = dict(dv)
