#pragma once

#include "tools/loggy/backend/session.h"
#include "tools/loggy/backend/store.h"
#include "tools/loggy/backend/undo.h"

#include "json11/json11.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace loggy {

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

struct FindSignalPaneState {
  int bus = -1;
  std::string address_hex;
  int min_size = 1;
  int max_size = 12;
  bool little_endian = true;
  bool is_signed = false;
  double factor = 1.0;
  double offset = 0.0;
  double target_value = 0.0;
  FindSignalCompare compare = FindSignalCompare::Equal;
  std::string status;
};

const char *find_signal_compare_token(FindSignalCompare compare);
const char *find_signal_compare_label(FindSignalCompare compare);
FindSignalCompare find_signal_compare_from_token(std::string_view token);
bool find_signal_compare_value(double value, FindSignalCompare compare, double target);

FindSignalPaneState parse_find_signal_pane_state(std::string_view state_json);
std::string find_signal_pane_state_json(const FindSignalPaneState &state);
FindSignalParams find_signal_params_from_state(const FindSignalPaneState &state, TimeRange range);

FindSignalJob make_find_signal_job(const Store &store, const FindSignalParams &params);
bool step_find_signal_job(FindSignalJob &job, size_t max_candidates);
std::vector<FindSignalResult> prepare_find_signal_candidates(const Store &store,
                                                             const FindSignalParams &params);
bool commit_find_signal_result(Session &session, std::string_view selection_group,
                               const FindSignalResult &result, std::string *error = nullptr);

void draw_find_signal_pane(Session &session, PaneInstance &pane);

}  // namespace loggy
