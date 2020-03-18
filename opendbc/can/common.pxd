# distutils: language = c++
#cython: language_level=3

from libc.stdint cimport uint32_t, uint64_t, uint16_t
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.unordered_set cimport unordered_set
from libcpp cimport bool


cdef extern from "common_dbc.h":
  ctypedef enum SignalType:
    DEFAULT,
    HONDA_CHECKSUM,
    HONDA_COUNTER,
    TOYOTA_CHECKSUM,
    PEDAL_CHECKSUM,
    PEDAL_COUNTER,
    VOLKSWAGEN_CHECKSUM,
    VOLKSWAGEN_COUNTER

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

  cdef struct SignalParseOptions:
    uint32_t address
    const char* name
    double default_value


  cdef struct MessageParseOptions:
    uint32_t address
    int check_frequency

  cdef struct SignalValue:
    uint32_t address
    uint16_t ts
    const char* name
    double value

  cdef struct SignalPackValue:
    const char * name
    double value


cdef extern from "common.h":
  cdef const DBC* dbc_lookup(const string);

  cdef cppclass CANParser:
    bool can_valid
    CANParser(int, string, vector[MessageParseOptions], vector[SignalParseOptions])
    void update_string(string, bool)
    vector[SignalValue] query_latest()

  cdef cppclass CANPacker:
   CANPacker(string)
   uint64_t pack(uint32_t, vector[SignalPackValue], int counter)
