#pragma once

#include "tools/loggy/backend/session.h"
#include "tools/loggy/backend/store.h"

#include "json11/json11.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace loggy {

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

struct FindBitsPaneState {
  std::string source = "0:47";
  int byte_idx = 0;
  int bit_idx = 0;
  int find_bus = 0;
  bool equal = true;
  int min_msgs = 0;
  std::string status;
};

FindBitsPaneState parse_find_bits_pane_state(std::string_view state_json);
std::string find_bits_pane_state_json(const FindBitsPaneState &state);
FindBitsParams find_bits_params_from_state(const FindBitsPaneState &state, TimeRange range);

std::vector<FindBitsEvent> collect_find_bits_events(const Store &store, const FindBitsParams &params);
std::vector<FindBitsRow> scan_find_bits_events(const std::vector<FindBitsEvent> &events,
                                               const FindBitsParams &params);
FindBitsJob make_find_bits_job(const Store &store, const FindBitsParams &params);
bool step_find_bits_job(FindBitsJob &job, size_t max_messages);
void activate_find_bits_row(Session &session, std::string_view selection_group,
                            const FindBitsRow &row, uint8_t bus);

void draw_find_bits_pane(Session &session, PaneInstance &pane);

}  // namespace loggy
