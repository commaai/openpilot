# distutils: language = c++
# cython: c_string_encoding=ascii, language_level=3

from libc.stdint cimport uint8_t
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.string cimport string

from .common cimport CANPacker as cpp_CANPacker
from .common cimport dbc_lookup, SignalPackValue, DBC


cdef class CANPacker:
  cdef:
    cpp_CANPacker *packer
    const DBC *dbc
    map[string, (int, int)] name_to_address_and_size
    map[int, int] address_to_size

  def __init__(self, dbc_name):
    self.dbc = dbc_lookup(dbc_name)
    if not self.dbc:
      raise RuntimeError(f"Can't lookup {dbc_name}")

    self.packer = new cpp_CANPacker(dbc_name)
    for i in range(self.dbc[0].msgs.size()):
      msg = self.dbc[0].msgs[i]
      self.name_to_address_and_size[string(msg.name)] = (msg.address, msg.size)
      self.address_to_size[msg.address] = msg.size

  cdef vector[uint8_t] pack(self, addr, values):
    cdef vector[SignalPackValue] values_thing
    values_thing.reserve(len(values))
    cdef SignalPackValue spv

    for name, value in values.iteritems():
      spv.name = name.encode("utf8")
      spv.value = value
      values_thing.push_back(spv)

    return self.packer.pack(addr, values_thing)

  cpdef make_can_msg(self, name_or_addr, bus, values):
    cdef int addr, size
    if type(name_or_addr) == int:
      addr = name_or_addr
      size = self.address_to_size[name_or_addr]
    else:
      addr, size = self.name_to_address_and_size[name_or_addr.encode("utf8")]

    cdef vector[uint8_t] val = self.pack(addr, values)
    return [addr, 0, (<char *>&val[0])[:size], bus]
