# distutils: language = c++
from libc.stdint cimport uint32_t, uint64_t, uint16_t
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.unordered_set cimport unordered_set
from libcpp cimport bool

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

ctypedef const DBC * (*dbc_lookup_func)(const char* dbc_name)
ctypedef void* (*can_init_with_vectors_func)(int bus, const char* dbc_name,
                vector[MessageParseOptions] message_options,
                vector[SignalParseOptions] signal_options,
                bool sendcan,
                const char* tcp_addr,
                int timeout)
ctypedef int (*can_update_func)(void* can, uint64_t sec, bool wait);
ctypedef size_t (*can_query_func)(void* can, uint64_t sec, bool *out_can_valid, size_t out_values_size, SignalValue* out_values);
ctypedef void (*can_query_vector_func)(void* can, uint64_t sec, bool *out_can_valid,  vector[SignalValue] &values)

cdef class CANParser:
  cdef:
    void *can
    const DBC *dbc
    dbc_lookup_func dbc_lookup
    can_init_with_vectors_func can_init_with_vectors
    can_update_func can_update
    can_query_vector_func can_query_vector
    map[string, uint32_t] msg_name_to_address
    map[uint32_t, string] address_to_msg_name
    vector[SignalValue] can_values
    bool test_mode_enabled
  cdef public:
    string dbc_name
    dict vl
    dict ts
    bool can_valid
    int can_invalid_cnt

  cdef unordered_set[uint32_t] update_vl(self, uint64_t sec)
