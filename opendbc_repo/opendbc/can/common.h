#pragma once

#include <map>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

#include "opendbc/can/logger.h"
#include "opendbc/can/common_dbc.h"

#define INFO printf
#define WARN printf
#define DEBUG(...)
//#define DEBUG printf

#define MAX_BAD_COUNTER 5
#define CAN_INVALID_CNT 5

// Car specific functions
unsigned int honda_checksum(uint32_t address, const Signal &sig, const std::vector<uint8_t> &d);
unsigned int toyota_checksum(uint32_t address, const Signal &sig, const std::vector<uint8_t> &d);
unsigned int subaru_checksum(uint32_t address, const Signal &sig, const std::vector<uint8_t> &d);
unsigned int chrysler_checksum(uint32_t address, const Signal &sig, const std::vector<uint8_t> &d);
unsigned int volkswagen_mqb_checksum(uint32_t address, const Signal &sig, const std::vector<uint8_t> &d);
unsigned int xor_checksum(uint32_t address, const Signal &sig, const std::vector<uint8_t> &d);
unsigned int hkg_can_fd_checksum(uint32_t address, const Signal &sig, const std::vector<uint8_t> &d);
unsigned int pedal_checksum(uint32_t address, const Signal &sig, const std::vector<uint8_t> &d);

struct CanFrame {
  long src;
  uint32_t address;
  std::vector<uint8_t> dat;
};

struct CanData {
  uint64_t nanos;
  std::vector<CanFrame> frames;
};

class MessageState {
public:
  std::string name;
  uint32_t address;
  unsigned int size;

  std::vector<Signal> parse_sigs;
  std::vector<double> vals;
  std::vector<std::vector<double>> all_vals;

  uint64_t last_seen_nanos;
  uint64_t check_threshold;

  uint8_t counter;
  uint8_t counter_fail;

  bool ignore_checksum = false;
  bool ignore_counter = false;

  bool parse(uint64_t nanos, const std::vector<uint8_t> &dat);
  bool update_counter_generic(int64_t v, int cnt_size);
};

class CANParser {
private:
  const int bus;
  const DBC *dbc = NULL;
  std::unordered_map<uint32_t, MessageState> message_states;

public:
  bool can_valid = false;
  bool bus_timeout = false;
  uint64_t first_nanos = 0;
  uint64_t last_nanos = 0;
  uint64_t last_nonempty_nanos = 0;
  uint64_t bus_timeout_threshold = 0;
  uint64_t can_invalid_cnt = CAN_INVALID_CNT;

  CANParser(int abus, const std::string& dbc_name,
            const std::vector<std::pair<uint32_t, int>> &messages);
  CANParser(int abus, const std::string& dbc_name, bool ignore_checksum, bool ignore_counter);
  void update(const std::vector<CanData> &can_data, std::vector<SignalValue> &vals);
  void query_latest(std::vector<SignalValue> &vals, uint64_t last_ts = 0);

protected:
  void UpdateCans(const CanData &can);
  void UpdateValid(uint64_t nanos);
};

class CANPacker {
private:
  const DBC *dbc = NULL;
  std::map<std::pair<uint32_t, std::string>, Signal> signal_lookup;
  std::map<uint32_t, uint32_t> counters;

public:
  CANPacker(const std::string& dbc_name);
  std::vector<uint8_t> pack(uint32_t address, const std::vector<SignalPackValue> &values);
  const Msg* lookup_message(uint32_t address);
};
