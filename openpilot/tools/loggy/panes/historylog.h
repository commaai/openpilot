#pragma once

#include "tools/loggy/backend/dbc/dbc.h"
#include "tools/loggy/panes/messages.h"

#include "json11/json11.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace loggy {

struct HistoryLogState {
  std::string filter;
  std::string compare_signal;
  std::string compare_op = ">";
  double compare_value = 0.0;
  bool compare_enabled = false;
  size_t max_rows = 1000;
  size_t page_size = 250;
  size_t page_index = 0;
  std::string export_path = "/tmp/loggy_history.csv";
  std::string export_status;
};

struct HistoryLogRow {
  double mono_time = 0.0;
  uint16_t bus_time = 0;
  size_t byte_count = 0;
  std::string data_hex;
  std::string decoded;
};

struct HistoryLogPage {
  std::vector<HistoryLogRow> rows;
  size_t total_rows = 0;
  size_t page_index = 0;
  size_t page_size = 250;
  size_t page_count = 1;
  bool truncated = false;
};

inline bool history_valid_compare_op(std::string_view op) {
  return op == ">" || op == "=" || op == "!=" || op == "<" || op == ">=" || op == "<=";
}

inline HistoryLogState parse_history_log_state(std::string_view state_json) {
  HistoryLogState state;
  std::string err;
  const json11::Json json = json11::Json::parse(std::string(state_json), err);
  if (!err.empty() || !json.is_object()) return state;
  if (json["filter"].is_string()) state.filter = json["filter"].string_value();
  if (json["compare_signal"].is_string()) state.compare_signal = json["compare_signal"].string_value();
  if (json["compare_op"].is_string() && history_valid_compare_op(json["compare_op"].string_value())) {
    state.compare_op = json["compare_op"].string_value();
  }
  if (json["compare_value"].is_number()) state.compare_value = json["compare_value"].number_value();
  if (json["compare_enabled"].is_bool()) state.compare_enabled = json["compare_enabled"].bool_value();
  if (json["max_rows"].is_number()) state.max_rows = static_cast<size_t>(std::clamp(json["max_rows"].int_value(), 16, 20000));
  if (json["page_size"].is_number()) state.page_size = static_cast<size_t>(std::clamp(json["page_size"].int_value(), 1, 5000));
  if (json["page_index"].is_number()) state.page_index = static_cast<size_t>(std::max(0, json["page_index"].int_value()));
  if (json["export_path"].is_string()) state.export_path = json["export_path"].string_value();
  if (json["export_status"].is_string()) state.export_status = json["export_status"].string_value();
  return state;
}

inline std::string history_log_state_json(const MessageId &id, const HistoryLogState &state) {
  return json11::Json(json11::Json::object{
    {"id", id.toString()},
    {"source", static_cast<int>(id.source)},
    {"address", static_cast<int>(id.address)},
    {"filter", state.filter},
    {"compare_signal", state.compare_signal},
    {"compare_op", history_valid_compare_op(state.compare_op) ? state.compare_op : std::string(">")},
    {"compare_value", state.compare_value},
    {"compare_enabled", state.compare_enabled},
    {"max_rows", static_cast<int>(state.max_rows)},
    {"page_size", static_cast<int>(state.page_size)},
    {"page_index", static_cast<int>(state.page_index)},
    {"export_path", state.export_path},
    {"export_status", state.export_status},
  }).dump();
}

inline std::string history_log_state_json(const HistoryLogState &state) {
  return history_log_state_json(kDefaultLoggyMessageId, state);
}

inline std::string history_lower_text(std::string_view text) {
  std::string out(text);
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}

inline bool history_text_matches_filter(std::string_view text, std::string_view filter) {
  return filter.empty() || history_lower_text(text).find(history_lower_text(filter)) != std::string::npos;
}

inline bool history_compare_values(double lhs, std::string_view op, double rhs) {
  if (op == ">") return lhs > rhs;
  if (op == "=") return lhs == rhs;
  if (op == "!=") return lhs != rhs;
  if (op == "<") return lhs < rhs;
  if (op == ">=") return lhs >= rhs;
  if (op == "<=") return lhs <= rhs;
  return true;
}

inline std::string history_hex_bytes(const std::vector<uint8_t> &bytes) {
  std::string out;
  out.reserve(bytes.size() * 3);
  for (size_t i = 0; i < bytes.size(); ++i) {
    char buf[4];
    std::snprintf(buf, sizeof(buf), "%02X", bytes[i]);
    if (!out.empty()) out.push_back(' ');
    out += buf;
  }
  return out;
}

inline std::string history_decoded_values(const Msg *msg, const std::vector<uint8_t> &data, size_t max_values = 4) {
  if (msg == nullptr || data.empty()) return {};
  std::string out;
  size_t count = 0;
  for (const Signal *sig : msg->getSignals()) {
    if (sig == nullptr) continue;
    double value = 0.0;
    if (!sig->getValue(data.data(), data.size(), &value)) continue;
    if (!out.empty()) out += ", ";
    out += sig->name + "=" + sig->formatValue(value);
    if (++count >= max_values) break;
  }
  return out;
}

inline bool history_matches_compare(const Msg *msg, const HistoryLogState &state, const std::vector<uint8_t> &data) {
  if (!state.compare_enabled || state.compare_signal.empty()) return true;
  if (msg == nullptr || data.empty()) return false;
  const Signal *sig = msg->sig(state.compare_signal);
  if (sig == nullptr) return false;
  double value = 0.0;
  if (!sig->getValue(data.data(), data.size(), &value)) return false;
  return history_compare_values(value, state.compare_op, state.compare_value);
}

inline HistoryLogPage prepare_history_log_page(const Store &store,
                                               const MessageId &id,
                                               TimeRange range,
                                               const HistoryLogState &state,
                                               const Msg *msg = nullptr) {
  const CanEventView view = store.canEvents(id, range);
  std::vector<HistoryLogRow> matches;
  matches.reserve(std::min(view.events.size(), state.max_rows));
  bool truncated = false;
  size_t event_index = 0;
  for (auto it = view.events.rbegin(); it != view.events.rend(); ++it) {
    const CanEvent &event = *it;
    ++event_index;
    if (!history_matches_compare(msg, state, event.data)) continue;

    HistoryLogRow row;
    row.mono_time = event.mono_time;
    row.bus_time = event.bus_time;
    row.byte_count = event.data.size();
    row.data_hex = history_hex_bytes(event.data);
    row.decoded = history_decoded_values(msg, event.data);
    const std::string searchable = row.data_hex + " " + row.decoded;
    if (!history_text_matches_filter(searchable, state.filter)) continue;
    matches.push_back(std::move(row));
    if (matches.size() >= state.max_rows) {
      truncated = event_index < view.events.size();
      break;
    }
  }

  HistoryLogPage page;
  page.total_rows = matches.size();
  page.page_size = std::clamp(state.page_size, static_cast<size_t>(1), static_cast<size_t>(5000));
  page.page_count = std::max(static_cast<size_t>(1), (matches.size() + page.page_size - 1) / page.page_size);
  page.page_index = std::min(state.page_index, page.page_count - 1);
  page.truncated = truncated;

  const size_t start = std::min(matches.size(), page.page_index * page.page_size);
  const size_t end = std::min(matches.size(), start + page.page_size);
  page.rows.assign(matches.begin() + static_cast<std::ptrdiff_t>(start), matches.begin() + static_cast<std::ptrdiff_t>(end));
  return page;
}

inline std::vector<HistoryLogRow> prepare_history_log_rows(const Store &store,
                                                           const MessageId &id,
                                                           TimeRange range,
                                                           const HistoryLogState &state,
                                                           const Msg *msg = nullptr) {
  return prepare_history_log_page(store, id, range, state, msg).rows;
}

inline std::vector<HistoryLogRow> prepare_history_log_rows_oldest_first(const Store &store,
                                                                        const MessageId &id,
                                                                        TimeRange range,
                                                                        const HistoryLogState &state,
                                                                        const Msg *msg = nullptr) {
  const CanEventView view = store.canEvents(id, range);
  std::vector<HistoryLogRow> rows;
  rows.reserve(std::min(view.events.size(), state.max_rows));
  for (const CanEvent &event : view.events) {
    if (!history_matches_compare(msg, state, event.data)) continue;
    HistoryLogRow row;
    row.mono_time = event.mono_time;
    row.bus_time = event.bus_time;
    row.byte_count = event.data.size();
    row.data_hex = history_hex_bytes(event.data);
    row.decoded = history_decoded_values(msg, event.data);
    const std::string searchable = row.data_hex + " " + row.decoded;
    if (!history_text_matches_filter(searchable, state.filter)) continue;
    rows.push_back(std::move(row));
    if (rows.size() >= state.max_rows) break;
  }
  return rows;
}

void draw_history_log_pane(Session &session, PaneInstance &pane);

}  // namespace loggy
