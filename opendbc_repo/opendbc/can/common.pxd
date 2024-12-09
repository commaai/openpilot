# distutils: language = c++
# cython: language_level=3

from libc.stdint cimport uint8_t, uint32_t, uint64_t
from libcpp cimport bool
from libcpp.pair cimport pair
from libcpp.set cimport set
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map


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
    FCA_GIORGIO_CHECKSUM,

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
    unordered_map[uint32_t, const Msg*] addr_to_msg
    unordered_map[string, const Msg*] name_to_msg

  cdef struct SignalPackValue:
    string name
    double value


cdef extern from "common.h":
  cdef const DBC* dbc_lookup(const string) except +

  cdef cppclass MessageState:
    vector[Signal] parse_sigs
    vector[double] vals
    vector[vector[double]] all_vals
    uint64_t last_seen_nanos

  cdef struct CanFrame:
    long src
    uint32_t address
    vector[uint8_t] dat

  cdef struct CanData:
    uint64_t nanos
    vector[CanFrame] frames

  cdef cppclass CANParser:
    bool can_valid
    bool bus_timeout
    CANParser(int, string, vector[pair[uint32_t, int]]) except +
    set[uint32_t] update(vector[CanData]&) except +
    MessageState *getMessageState(uint32_t address)

  cdef cppclass CANPacker:
   CANPacker(string)
   vector[uint8_t] pack(uint32_t, vector[SignalPackValue]&)
