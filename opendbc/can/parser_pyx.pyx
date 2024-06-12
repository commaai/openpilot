# distutils: language = c++
# cython: c_string_encoding=ascii, language_level=3

from cython.operator cimport dereference as deref, preincrement as preinc
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stdint cimport uint32_t

from .common cimport CANParser as cpp_CANParser
from .common cimport dbc_lookup, SignalValue, DBC

import numbers
from collections import defaultdict


cdef class CANParser:
  cdef:
    cpp_CANParser *can
    const DBC *dbc
    vector[SignalValue] can_values
    vector[uint32_t] addresses

  cdef readonly:
    dict vl
    dict vl_all
    dict ts_nanos
    string dbc_name

  def __init__(self, dbc_name, messages, bus=0):
    self.dbc_name = dbc_name
    self.dbc = dbc_lookup(dbc_name)
    if not self.dbc:
      raise RuntimeError(f"Can't find DBC: {dbc_name}")

    self.vl = {}
    self.vl_all = {}
    self.ts_nanos = {}

    # Convert message names into addresses and check existence in DBC
    cdef vector[pair[uint32_t, int]] message_v
    for i in range(len(messages)):
      c = messages[i]
      try:
        m = self.dbc.addr_to_msg.at(c[0]) if isinstance(c[0], numbers.Number) else self.dbc.name_to_msg.at(c[0])
      except IndexError:
        raise RuntimeError(f"could not find message {repr(c[0])} in DBC {self.dbc_name}")

      address = m.address
      message_v.push_back((address, c[1]))
      self.addresses.push_back(address)

      name = m.name.decode("utf8")
      self.vl[address] = {}
      self.vl[name] = self.vl[address]
      self.vl_all[address] = defaultdict(list)
      self.vl_all[name] = self.vl_all[address]
      self.ts_nanos[address] = {}
      self.ts_nanos[name] = self.ts_nanos[address]

    self.can = new cpp_CANParser(bus, dbc_name, message_v)
    self.update_strings([])

  def __dealloc__(self):
    if self.can:
      del self.can

  def update_strings(self, strings, sendcan=False):
    for address in self.addresses:
      self.vl_all[address].clear()

    cdef vector[SignalValue] new_vals
    cur_address = -1
    vl = {}
    vl_all = {}
    ts_nanos = {}
    updated_addrs = set()

    self.can.update_strings(strings, new_vals, sendcan)
    cdef vector[SignalValue].iterator it = new_vals.begin()
    cdef SignalValue* cv
    while it != new_vals.end():
      cv = &deref(it)

      # Check if the address has changed
      if cv.address != cur_address:
        cur_address = cv.address
        vl = self.vl[cur_address]
        vl_all = self.vl_all[cur_address]
        ts_nanos = self.ts_nanos[cur_address]
        updated_addrs.add(cur_address)

      # Cast char * directly to unicode
      cv_name = <unicode>cv.name
      vl[cv_name] = cv.value
      vl_all[cv_name] = cv.all_values
      ts_nanos[cv_name] = cv.ts_nanos
      preinc(it)

    return updated_addrs

  @property
  def can_valid(self):
    return self.can.can_valid

  @property
  def bus_timeout(self):
    return self.can.bus_timeout


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

    dv = defaultdict(dict)

    for i in range(self.dbc[0].vals.size()):
      val = self.dbc[0].vals[i]

      sgname = val.name.decode("utf8")
      def_val = val.def_val.decode("utf8")
      address = val.address
      try:
        m = self.dbc.addr_to_msg.at(address)
      except IndexError:
        raise KeyError(address)
      msgname = m.name.decode("utf-8")

      # separate definition/value pairs
      def_val = def_val.split()
      values = [int(v) for v in def_val[::2]]
      defs = def_val[1::2]

      # two ways to lookup: address or msg name
      dv[address][sgname] = dict(zip(values, defs))
      dv[msgname][sgname] = dv[address][sgname]

    self.dv = dict(dv)
