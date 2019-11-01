#ifndef SELFDRIVE_CAN_COMMON_H
#define SELFDRIVE_CAN_COMMON_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <capnp/serialize.h>
#include "cereal/gen/cpp/log.capnp.h"

#define MAX_BAD_COUNTER 5
#define ARRAYSIZE(x) (sizeof(x)/sizeof(x[0]))

// Helper functions
unsigned int honda_checksum(unsigned int address, uint64_t d, int l);
unsigned int toyota_checksum(unsigned int address, uint64_t d, int l);
void init_crc_lookup_tables();
unsigned int volkswagen_crc(unsigned int address, uint64_t d, int l);
unsigned int pedal_checksum(uint64_t d, int l);
uint64_t read_u64_be(const uint8_t* v);
uint64_t read_u64_le(const uint8_t* v);

struct SignalPackValue {
  const char* name;
  double value;
};

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
  uint16_t ts;
  const char* name;
  double value;
};


enum SignalType {
  DEFAULT,
  HONDA_CHECKSUM,
  HONDA_COUNTER,
  TOYOTA_CHECKSUM,
  PEDAL_CHECKSUM,
  PEDAL_COUNTER,
  VOLKSWAGEN_CHECKSUM,
  VOLKSWAGEN_COUNTER,
};

struct Signal {
  const char* name;
  int b1, b2, bo;
  bool is_signed;
  double factor, offset;
  bool is_little_endian;
  SignalType type;
};

struct Msg {
  const char* name;
  uint32_t address;
  unsigned int size;
  size_t num_sigs;
  const Signal *sigs;
};

struct Val {
  const char* name;
  uint32_t address;
  const char* def_val;
  const Signal *sigs;
};

struct DBC {
  const char* name;
  size_t num_msgs;
  const Msg *msgs;
  const Val *vals;
  size_t num_vals;
};


class MessageState {
public:
  uint32_t address;
  unsigned int size;

  std::vector<Signal> parse_sigs;
  std::vector<double> vals;

  uint16_t ts;
  uint64_t seen;
  uint64_t check_threshold;

  uint8_t counter;
  uint8_t counter_fail;

  bool parse(uint64_t sec, uint16_t ts_, uint8_t * dat);
  bool update_counter_generic(int64_t v, int cnt_size);
};

class CANParser {
private:
  const int bus;

  const DBC *dbc = NULL;
  std::unordered_map<uint32_t, MessageState> message_states;

public:
  bool can_valid = false;
  uint64_t last_sec = 0;

  CANParser(int abus, const std::string& dbc_name,
            const std::vector<MessageParseOptions> &options,
            const std::vector<SignalParseOptions> &sigoptions);
  void UpdateCans(uint64_t sec, const capnp::List<cereal::CanData>::Reader& cans);
  void UpdateValid(uint64_t sec);
  void update_string(std::string data);
  std::vector<SignalValue> query_latest();

};




const DBC* dbc_lookup(const std::string& dbc_name);

void dbc_register(const DBC* dbc);

#define dbc_init(dbc) \
static void __attribute__((constructor)) do_dbc_init_ ## dbc(void) { \
  dbc_register(&dbc); \
}

#endif
