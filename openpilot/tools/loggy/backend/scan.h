#pragma once

#include "tools/loggy/backend/store.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace loggy {

// -- Find Bits: correlate one source bit against every other message's bits over time. --

struct FindBitsParams {
  TimeRange range;
  uint8_t source_bus = 0;
  uint32_t source_address = 0;
  int byte_idx = 0;
  int bit_idx = 0;
  uint8_t find_bus = 0;
  bool equal = true;
  int min_msgs = 0;
  size_t max_rows = 512;
};

struct FindBitsEvent {
  double mono_time = 0.0;
  uint8_t source_value = 0;
  MessageId id;
  std::vector<uint8_t> data;
};

struct FindBitsRow {
  uint32_t address = 0;
  uint32_t byte_idx = 0;
  uint32_t bit_idx = 0;
  uint32_t mismatches = 0;
  uint32_t total = 0;
  float percent = 0.0f;
};

struct FindBitsJob {
  const Store *store = nullptr;
  FindBitsParams params;
  std::vector<MessageId> ids;
  size_t id_index = 0;
  bool done = true;
  std::vector<FindBitsEvent> events;
  std::vector<FindBitsRow> rows;
};

uint8_t bit_value_at(const std::vector<uint8_t> &data, int byte_idx, int bit_idx);
std::vector<FindBitsRow> scan_find_bits_events(const std::vector<FindBitsEvent> &events, const FindBitsParams &params);
FindBitsJob make_find_bits_job(const Store &store, const FindBitsParams &params);
bool step_find_bits_job(FindBitsJob &job, size_t max_messages);

// -- Find Signal: brute-force scan of start-bit/size candidates against a value comparator. --

enum class FindSignalCompare {
  Any,
  Equal,
  NotEqual,
  Greater,
  GreaterEqual,
  Less,
  LessEqual,
};

struct FindSignalParams {
  TimeRange range;
  std::vector<int> buses;
  std::vector<uint32_t> addresses;
  int min_size = 1;
  int max_size = 16;
  bool little_endian = true;
  bool is_signed = false;
  double factor = 1.0;
  double offset = 0.0;
  double target_value = 0.0;
  FindSignalCompare compare = FindSignalCompare::Any;
  size_t max_results = 512;
};

struct FindSignalResult {
  MessageId id;
  Signal sig;
  uint32_t msg_size = 0;
  double mono_time = 0.0;
  std::vector<std::pair<double, double>> matches;
};

struct FindSignalJob {
  const Store *store = nullptr;
  FindSignalParams params;
  std::vector<MessageId> ids;
  size_t id_index = 0;
  int start_bit = 0;
  int size = 1;
  bool done = true;
  std::vector<FindSignalResult> results;
};

bool find_signal_compare_value(double value, FindSignalCompare compare, double target);
FindSignalJob make_find_signal_job(const Store &store, const FindSignalParams &params);
bool step_find_signal_job(FindSignalJob &job, size_t max_candidates);

}  // namespace loggy
