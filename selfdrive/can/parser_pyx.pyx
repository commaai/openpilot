# distutils: language = c++
from posix.dlfcn cimport dlopen, dlsym, RTLD_LAZY

from libcpp cimport bool
import os
import numbers

cdef int CAN_INVALID_CNT = 5

cdef class CANParser:
  def __init__(self, dbc_name, signals, checks=None, bus=0, sendcan=False, tcp_addr="", timeout=-1):
    self.test_mode_enabled = False
    can_dir = os.path.dirname(os.path.abspath(__file__))
    libdbc_fn = os.path.join(can_dir, "libdbc.so")

    cdef void *libdbc = dlopen(libdbc_fn, RTLD_LAZY)
    self.can_init_with_vectors = <can_init_with_vectors_func>dlsym(libdbc, 'can_init_with_vectors')
    self.dbc_lookup = <dbc_lookup_func>dlsym(libdbc, 'dbc_lookup')
    self.can_update = <can_update_func>dlsym(libdbc, 'can_update')
    self.can_update_string = <can_update_string_func>dlsym(libdbc, 'can_update_string')
    self.can_query_vector = <can_query_vector_func>dlsym(libdbc, 'can_query_vector')
    if checks is None:
      checks = []

    self.can_valid = True
    self.dbc_name = dbc_name
    self.dbc = self.dbc_lookup(dbc_name)
    self.vl = {}
    self.ts = {}

    self.can_invalid_cnt = CAN_INVALID_CNT

    num_msgs = self.dbc[0].num_msgs
    for i in range(num_msgs):
      msg = self.dbc[0].msgs[i]
      self.msg_name_to_address[string(msg.name)] = msg.address
      self.address_to_msg_name[msg.address] = string(msg.name)
      self.vl[msg.address] = {}
      self.vl[str(msg.name)] = {}
      self.ts[msg.address] = {}
      self.ts[str(msg.name)] = {}

    # Convert message names into addresses
    for i in range(len(signals)):
      s = signals[i]
      if not isinstance(s[1], numbers.Number):
        s = (s[0], self.msg_name_to_address[s[1]], s[2])
        signals[i] = s

    for i in range(len(checks)):
      c = checks[i]
      if not isinstance(c[0], numbers.Number):
        c = (self.msg_name_to_address[c[0]], c[1])
        checks[i] = c

    cdef vector[SignalParseOptions] signal_options_v
    cdef SignalParseOptions spo
    for sig_name, sig_address, sig_default in signals:
      spo.address = sig_address
      spo.name = sig_name
      spo.default_value = sig_default
      signal_options_v.push_back(spo)

    message_options = dict((address, 0) for _, address, _ in signals)
    message_options.update(dict(checks))

    cdef vector[MessageParseOptions] message_options_v
    cdef MessageParseOptions mpo
    for msg_address, freq in message_options.items():
      mpo.address = msg_address
      mpo.check_frequency = freq
      message_options_v.push_back(mpo)

    self.can = self.can_init_with_vectors(bus, dbc_name, message_options_v, signal_options_v, sendcan, tcp_addr, timeout)
    self.update_vl(0)

  cdef unordered_set[uint32_t] update_vl(self, uint64_t sec):
    cdef string sig_name
    cdef unordered_set[uint32_t] updated_val
    cdef bool valid = False

    self.can_query_vector(self.can, sec, &valid, self.can_values)

    # Update invalid flag
    self.can_invalid_cnt += 1
    if valid:
        self.can_invalid_cnt = 0
    self.can_valid = self.can_invalid_cnt < CAN_INVALID_CNT


    for cv in self.can_values:
      self.vl[cv.address][string(cv.name)] = cv.value
      self.ts[cv.address][string(cv.name)] = cv.ts

      sig_name = self.address_to_msg_name[cv.address]
      self.vl[sig_name][string(cv.name)] = cv.value
      self.ts[sig_name][string(cv.name)] = cv.ts

      updated_val.insert(cv.address)

    return updated_val

  def update_string(self, uint64_t sec, dat):
    self.can_update_string(self.can, sec, dat, len(dat))
    return self.update_vl(sec)

  def update_strings(self, uint64_t sec, strings):
    updated_vals = set()

    for s in strings:
      updated_val = self.update_string(sec, s)
      updated_vals.update(updated_val)

    return updated_vals

  def update(self, uint64_t sec, bool wait):
    r = (self.can_update(self.can, sec, wait) >= 0)
    updated_val = self.update_vl(sec)
    return r, updated_val
