# distutils: language = c++
# cython: c_string_encoding=ascii, language_level=3

from libc.stdint cimport uint32_t, uint64_t
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp cimport bool
from posix.dlfcn cimport dlopen, dlsym, RTLD_LAZY

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
    num_msgs = self.dbc[0].num_msgs
    for i in range(num_msgs):
      msg = self.dbc[0].msgs[i]
      self.name_to_address_and_size[string(msg.name)] = (msg.address, msg.size)
      self.address_to_size[msg.address] = msg.size

  cdef uint64_t pack(self, addr, values, counter):
    cdef vector[SignalPackValue] values_thing
    cdef SignalPackValue spv

    names = []

    for name, value in values.iteritems():
      n = name.encode('utf8')
      names.append(n) # TODO: find better way to keep reference to temp string around

      spv.name = n
      spv.value = value
      values_thing.push_back(spv)

    return self.packer.pack(addr, values_thing, counter)

  cdef inline uint64_t ReverseBytes(self, uint64_t x):
    return (((x & 0xff00000000000000ull) >> 56) |
           ((x & 0x00ff000000000000ull) >> 40) |
           ((x & 0x0000ff0000000000ull) >> 24) |
           ((x & 0x000000ff00000000ull) >> 8) |
           ((x & 0x00000000ff000000ull) << 8) |
           ((x & 0x0000000000ff0000ull) << 24) |
           ((x & 0x000000000000ff00ull) << 40) |
           ((x & 0x00000000000000ffull) << 56))

  cpdef make_can_msg(self, name_or_addr, bus, values, counter=-1):
    cdef int addr, size
    if type(name_or_addr) == int:
      addr = name_or_addr
      size = self.address_to_size[name_or_addr]
    else:
      addr, size = self.name_to_address_and_size[name_or_addr.encode('utf8')]
    cdef uint64_t val = self.pack(addr, values, counter)
    val = self.ReverseBytes(val)
    return [addr, 0, (<char *>&val)[:size], bus]
