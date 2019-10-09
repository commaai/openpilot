# distutils: language = c++
# cython: c_string_encoding=ascii, language_level=3

from libc.stdint cimport uint32_t, uint64_t
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp cimport bool
from posix.dlfcn cimport dlopen, dlsym, RTLD_LAZY
import os
import subprocess

cdef struct SignalPackValue:
  const char* name
  double value

ctypedef enum SignalType:
  DEFAULT,
  HONDA_CHECKSUM,
  HONDA_COUNTER,
  TOYOTA_CHECKSUM,
  PEDAL_CHECKSUM,
  PEDAL_COUNTER

cdef struct Signal:
  const char* name
  int b1, b2, bo
  bool is_signed
  double factor, offset
  SignalType type



cdef struct Msg:
  const char* name
  uint32_t address
  unsigned int size
  size_t num_sigs
  const Signal *sigs

cdef struct Val:
  const char* name
  uint32_t address
  const char* def_val
  const Signal *sigs

cdef struct DBC:
  const char* name
  size_t num_msgs
  const Msg *msgs
  const Val *vals
  size_t num_vals

ctypedef void * (*canpack_init_func)(const char* dbc_name)
ctypedef uint64_t (*canpack_pack_vector_func)(void* inst, uint32_t address, const vector[SignalPackValue] &signals, int counter)
ctypedef const DBC * (*dbc_lookup_func)(const char* dbc_name)


cdef class CANPacker():
  cdef void *packer
  cdef const DBC *dbc
  cdef map[string, (int, int)] name_to_address_and_size
  cdef map[int, int] address_to_size
  cdef canpack_init_func canpack_init
  cdef canpack_pack_vector_func canpack_pack_vector
  cdef dbc_lookup_func dbc_lookup

  def __init__(self, dbc_name):
    can_dir = os.path.dirname(os.path.abspath(__file__))
    libdbc_fn = os.path.join(can_dir, "libdbc.so")
    libdbc_fn = str(libdbc_fn).encode('utf8')
    subprocess.check_call(["make"], cwd=can_dir)

    cdef void *libdbc = dlopen(libdbc_fn, RTLD_LAZY)
    self.canpack_init = <canpack_init_func>dlsym(libdbc, 'canpack_init')
    self.canpack_pack_vector = <canpack_pack_vector_func>dlsym(libdbc, 'canpack_pack_vector')
    self.dbc_lookup = <dbc_lookup_func>dlsym(libdbc, 'dbc_lookup')

    self.packer = self.canpack_init(dbc_name)
    self.dbc = self.dbc_lookup(dbc_name)
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
      names.append(n) # TODO: find better way to keep reference to temp string arround

      spv.name = n
      spv.value = value
      values_thing.push_back(spv)

    return self.canpack_pack_vector(self.packer, addr, values_thing, counter)

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
