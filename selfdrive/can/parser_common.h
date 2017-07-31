#ifndef PARSER_COMMON_H
#define PARSER_COMMON_H

#include <cstddef>
#include <cstdint>

#define ARRAYSIZE(x) (sizeof(x)/sizeof(x[0]))



struct SignalParseOptions {
  uint32_t address;
  const char* name;
  double default_value;
};

struct MessageParseOptions {
  uint32_t address;
  int check_frequency;
};

struct SignalValue {
  uint32_t address;
  const char* name;
  double value;
};


enum SignalType {
  DEFAULT,
  HONDA_CHECKSUM,
  HONDA_COUNTER,
};

struct Signal {
  const char* name;
  int b1, b2, bo;
  bool is_signed;
  double factor, offset;
  SignalType type;
};

struct Msg {
  const char* name;
  uint32_t address;
  size_t num_sigs;
  const Signal *sigs;
};

struct DBC {
  const char* name;
  size_t num_msgs;
  const Msg *msgs;
};

void dbc_register(const DBC* dbc);

#define dbc_init(dbc) \
static void __attribute__((constructor)) do_dbc_init_ ## dbc(void) { \
  dbc_register(&dbc); \
}

#endif
