# distutils: language = c++
# cython: c_string_encoding=ascii, language_level=3

from cython.operator cimport dereference as deref, preincrement as preinc
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from libc.stdint cimport uint32_t
from libcpp.map cimport map

from .common cimport CANParser as cpp_CANParser
from .common cimport SignalParseOptions, MessageParseOptions, dbc_lookup, SignalValue, DBC

import numbers
from collections import defaultdict


cdef class CANParser:
  cdef:
    cpp_CANParser *can
    const DBC *dbc
    map[uint32_t, string] address_to_msg_name
    vector[SignalValue] can_values

  cdef readonly:
    dict vl
    dict vl_all
    dict ts_nanos
    string dbc_name

  def __init__(self, dbc_name, signals, checks=None, bus=0, enforce_checks=True):
    if checks is None:
      checks = []

    self.dbc_name = dbc_name
    self.dbc = dbc_lookup(dbc_name)
    if not self.dbc:
      raise RuntimeError(f"Can't find DBC: {dbc_name}")

    self.vl = {}
    self.vl_all = {}
    self.ts_nanos = {}
    msg_name_to_address = {}
    msg_address_to_signals = {}

    for i in range(self.dbc[0].msgs.size()):
      msg = self.dbc[0].msgs[i]
      name = msg.name.decode("utf8")

      msg_name_to_address[name] = msg.address
      msg_address_to_signals[msg.address] = set()
      for sig in msg.sigs:
        msg_address_to_signals[msg.address].add(sig.name.decode("utf8"))

      self.address_to_msg_name[msg.address] = name
      self.vl[msg.address] = {}
      self.vl[name] = self.vl[msg.address]
      self.vl_all[msg.address] = {}
      self.vl_all[name] = self.vl_all[msg.address]
      self.ts_nanos[msg.address] = {}
      self.ts_nanos[name] = self.ts_nanos[msg.address]

    # Convert message names into addresses
    for i in range(len(signals)):
      s = signals[i]
      address = s[1] if isinstance(s[1], numbers.Number) else msg_name_to_address.get(s[1])
      if address not in msg_address_to_signals:
        raise RuntimeError(f"could not find message {repr(s[1])} in DBC {self.dbc_name}")
      if s[0] not in msg_address_to_signals[address]:
        raise RuntimeError(f"could not find signal {repr(s[0])} in {repr(s[1])}, DBC {self.dbc_name}")

      signals[i] = (s[0], address)

    for i in range(len(checks)):
      c = checks[i]
      if not isinstance(c[0], numbers.Number):
        if c[0] not in msg_name_to_address:
          print(msg_name_to_address)
          raise RuntimeError(f"could not find message {repr(c[0])} in DBC {self.dbc_name}")
        c = (msg_name_to_address[c[0]], c[1])
        checks[i] = c

    if enforce_checks:
      checked_addrs = {c[0] for c in checks}
      signal_addrs = {s[1] for s in signals}
      unchecked = signal_addrs - checked_addrs
      if len(unchecked):
        err_msg = ", ".join(f"{self.address_to_msg_name[addr].decode()} ({hex(addr)})" for addr in unchecked)
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
    self.update_strings([])

  def update_strings(self, strings, sendcan=False):
    for v in self.vl_all.values():
      for l in v.values():  # no-cython-lint
        l.clear()

    cdef vector[SignalValue] new_vals
    cdef unordered_set[uint32_t] updated_addrs

    self.can.update_strings(strings, new_vals, sendcan)
    cdef vector[SignalValue].iterator it = new_vals.begin()
    cdef SignalValue* cv
    while it != new_vals.end():
      cv = &deref(it)
      # Cast char * directly to unicode
      cv_name = <unicode>cv.name
      self.vl[cv.address][cv_name] = cv.value
      self.vl_all[cv.address][cv_name] = cv.all_values
      self.ts_nanos[cv.address][cv_name] = cv.ts_nanos
      updated_addrs.insert(cv.address)
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

    address_to_msg_name = {}

    for i in range(self.dbc[0].msgs.size()):
      msg = self.dbc[0].msgs[i]
      name = msg.name.decode("utf8")
      address = msg.address
      address_to_msg_name[address] = name

    dv = defaultdict(dict)

    for i in range(self.dbc[0].vals.size()):
      val = self.dbc[0].vals[i]

      sgname = val.name.decode("utf8")
      def_val = val.def_val.decode("utf8")
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
