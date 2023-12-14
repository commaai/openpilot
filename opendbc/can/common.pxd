# distutils: language = c++
# cython: language_level=3

from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t
from libcpp cimport bool
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.vector cimport vector


ctypedef unsigned int (*calc_checksum_type)(uint32_t, const Signal&, const vector[uint8_t] &)

cdef extern from "common_dbc.h":
  ctypedef enum SignalType:
    DEFAULT,
    COUNTER,
    HONDA_CHECKSUM,
    TOYOTA_CHECKSUM,
    PEDAL_CHECKSUM,
    VOLKSWAGEN_MQB_CHECKSUM,
    XOR_CHECKSUM,
    SUBARU_CHECKSUM,
    CHRYSLER_CHECKSUM
    HKG_CAN_FD_CHECKSUM,

  cdef struct Signal:
    string name
    int start_bit, msb, lsb, size
    bool is_signed
    double factor, offset
    bool is_little_endian
    SignalType type
    calc_checksum_type calc_checksum

  cdef struct Msg:
    string name
    uint32_t address
    unsigned int size
    vector[Signal] sigs

  cdef struct Val:
    string name
    uint32_t address
    string def_val
    vector[Signal] sigs

  cdef struct DBC:
    string name
    vector[Msg] msgs
    vector[Val] vals

  cdef struct SignalValue:
    uint32_t address
    uint64_t ts_nanos
    string name
    double value
    vector[double] all_values

  cdef struct SignalPackValue:
    string name
    double value


cdef extern from "common.h":
  cdef const DBC* dbc_lookup(const string) except +

  cdef cppclass CANParser:
    bool can_valid
    bool bus_timeout
    CANParser(int, string, vector[pair[uint32_t, int]]) except +
    void update_strings(vector[string]&, vector[SignalValue]&, bool) except +

  cdef cppclass CANPacker:
   CANPacker(string)
   vector[uint8_t] pack(uint32_t, vector[SignalPackValue]&)
