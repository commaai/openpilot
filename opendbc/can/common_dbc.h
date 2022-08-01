#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#define ARRAYSIZE(x) (sizeof(x)/sizeof(x[0]))

struct SignalPackValue {
  std::string name;
  double value;
};

struct SignalParseOptions {
  uint32_t address;
  std::string name;
};

struct MessageParseOptions {
  uint32_t address;
  int check_frequency;
};

struct SignalValue {
  uint32_t address;
  std::string name;
  double value;  // latest value
  std::vector<double> all_values;  // all values from this cycle
};

enum SignalType {
  DEFAULT,
  COUNTER,
  HONDA_CHECKSUM,
  TOYOTA_CHECKSUM,
  PEDAL_CHECKSUM,
  VOLKSWAGEN_MQB_CHECKSUM,
  XOR_CHECKSUM,
  SUBARU_CHECKSUM,
  CHRYSLER_CHECKSUM,
  HKG_CAN_FD_CHECKSUM,
};

struct Signal {
  std::string name;
  int start_bit, msb, lsb, size;
  bool is_signed;
  double factor, offset;
  bool is_little_endian;
  SignalType type;
  unsigned int (*calc_checksum)(uint32_t address, const Signal &sig, const std::vector<uint8_t> &d);
};

struct Msg {
  std::string name;
  uint32_t address;
  unsigned int size;
  std::vector<Signal> sigs;
};

struct Val {
  std::string name;
  uint32_t address;
  std::string def_val;
  std::vector<Signal> sigs;
};

struct DBC {
  std::string name;
  std::vector<Msg> msgs;
  std::vector<Val> vals;
};

DBC* dbc_parse(const std::string& dbc_path);
const DBC* dbc_lookup(const std::string& dbc_name);
std::vector<std::string> get_dbc_names();
