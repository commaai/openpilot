#pragma once

#include "tools/loggy/backend/store.h"
#include "tools/loggy/shell/pane.h"
#include "tools/loggy/shell/transport.h"

#include "json11/json11.hpp"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace loggy {

inline constexpr MessageId kDefaultLoggyMessageId{.source = 0, .address = 0x123};

struct MessageSummary {
  MessageId id;
  size_t count = 0;
  double first_time = 0.0;
  double last_time = 0.0;
  double frequency_hz = 0.0;
  std::vector<uint8_t> latest_data;
  CoverageInfo coverage;
};

struct MessageTableState {
  std::string filter;
  int bus_filter = -1;
  size_t max_rows = 500;
};

struct MessageTableRow {
  MessageId id;
  MessageSummary summary;
};

inline bool parse_u32_value(std::string_view text, int base, uint32_t *out) {
  if (text.empty() || out == nullptr) return false;
  std::string copy(text);
  char *end = nullptr;
  errno = 0;
  const unsigned long value = std::strtoul(copy.c_str(), &end, base);
  if (end != copy.c_str() + copy.size() || errno == ERANGE || value > std::numeric_limits<uint32_t>::max()) return false;
  *out = static_cast<uint32_t>(value);
  return true;
}

inline bool parse_message_id_text(std::string_view text, MessageId *out) {
  if (out == nullptr || text.empty()) return false;
  const size_t colon = text.find(':');
  if (colon == std::string_view::npos) {
    uint32_t address = 0;
    if (!parse_u32_value(text, 16, &address)) return false;
    *out = MessageId{.source = 0, .address = address};
    return true;
  }

  uint32_t source = 0;
  uint32_t address = 0;
  if (!parse_u32_value(text.substr(0, colon), 10, &source)) return false;
  const std::string_view addr_text = text.substr(colon + 1);
  if (!parse_u32_value(addr_text, 16, &address)) return false;
  *out = MessageId{.source = static_cast<uint8_t>(std::min<uint32_t>(source, 255)), .address = address};
  return true;
}

inline MessageId parse_message_id_state(std::string_view state_json,
                                        std::optional<MessageId> selection = std::nullopt,
                                        MessageId fallback = kDefaultLoggyMessageId) {
  if (selection.has_value()) return *selection;

  std::string err;
  const json11::Json state = json11::Json::parse(std::string(state_json), err);
  MessageId id = fallback;
  if (!err.empty()) {
    parse_message_id_text(state_json, &id);
    return id;
  }
  if (state.is_string()) {
    parse_message_id_text(state.string_value(), &id);
    return id;
  }
  if (!state.is_object()) return id;

  const json11::Json &text_id = state["id"].is_string() ? state["id"]
                            : state["message_id"].is_string() ? state["message_id"]
                            : state["selected_id"];
  if (text_id.is_string() && parse_message_id_text(text_id.string_value(), &id)) return id;

  const json11::Json &source_json = state["source"].is_number() ? state["source"] : state["bus"];
  const json11::Json &addr_json = state["address"].is_number() || state["address"].is_string() ? state["address"] : state["addr"];
  if (source_json.is_number()) id.source = static_cast<uint8_t>(std::clamp(source_json.int_value(), 0, 255));
  if (addr_json.is_number()) {
    id.address = static_cast<uint32_t>(std::max(0, addr_json.int_value()));
  } else if (addr_json.is_string()) {
    uint32_t address = id.address;
    if (parse_u32_value(addr_json.string_value(), 16, &address)) id.address = address;
  }
  return id;
}

inline MessageId initial_message_id_for_store(const Store &store, std::string_view state_json,
                                              std::optional<MessageId> selection = std::nullopt) {
  if (selection.has_value()) return *selection;
  const std::vector<MessageId> ids = store.canMessageIds();
  const MessageId fallback = ids.empty() ? kDefaultLoggyMessageId : ids.front();
  return parse_message_id_state(state_json, std::nullopt, fallback);
}

inline std::string message_id_state_json(const MessageId &id) {
  return json11::Json(json11::Json::object{
    {"id", id.toString()},
    {"source", static_cast<int>(id.source)},
    {"address", static_cast<int>(id.address)},
  }).dump();
}

inline MessageTableState parse_message_table_state(std::string_view state_json) {
  MessageTableState state;
  std::string err;
  const json11::Json json = json11::Json::parse(std::string(state_json), err);
  if (!err.empty() || !json.is_object()) return state;
  if (json["filter"].is_string()) state.filter = json["filter"].string_value();
  if (json["bus_filter"].is_number()) state.bus_filter = std::clamp(json["bus_filter"].int_value(), -1, 255);
  if (json["max_rows"].is_number()) state.max_rows = static_cast<size_t>(std::clamp(json["max_rows"].int_value(), 1, 5000));
  return state;
}

inline std::string message_table_state_json(const MessageId &id, const MessageTableState &state) {
  return json11::Json(json11::Json::object{
    {"id", id.toString()},
    {"source", static_cast<int>(id.source)},
    {"address", static_cast<int>(id.address)},
    {"filter", state.filter},
    {"bus_filter", state.bus_filter},
    {"max_rows", static_cast<int>(state.max_rows)},
  }).dump();
}

inline MessageSummary summarize_message_events(const Store &store, const MessageId &id, TimeRange range) {
  MessageSummary summary;
  summary.id = id;
  const CanSummaryView view = store.canEventSummary(id, range);
  summary.coverage = view.coverage;
  summary.count = view.count;
  if (view.count == 0) return summary;

  summary.first_time = view.first_time;
  summary.last_time = view.last_time;
  summary.latest_data = view.latest_data;
  if (summary.count > 1 && summary.last_time > summary.first_time) {
    summary.frequency_hz = static_cast<double>(summary.count - 1) / (summary.last_time - summary.first_time);
  }
  return summary;
}

inline std::string lower_message_text(std::string_view text) {
  std::string out(text);
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}

inline bool message_id_matches_filter(const MessageId &id, std::string_view filter) {
  if (filter.empty()) return true;
  const std::string needle = lower_message_text(filter);
  const std::string id_text = lower_message_text(id.toString());
  char hex_buf[32];
  std::snprintf(hex_buf, sizeof(hex_buf), "%X", id.address);
  const std::string hex_text = lower_message_text(hex_buf);
  char prefixed_hex_buf[36];
  std::snprintf(prefixed_hex_buf, sizeof(prefixed_hex_buf), "0x%X", id.address);
  const std::string prefixed_hex_text = lower_message_text(prefixed_hex_buf);
  return id_text.find(needle) != std::string::npos
      || hex_text.find(needle) != std::string::npos
      || prefixed_hex_text.find(needle) != std::string::npos;
}

inline std::vector<MessageTableRow> prepare_message_table_rows(const Store &store,
                                                               TimeRange range,
                                                               const MessageTableState &state) {
  const std::vector<MessageId> ids = store.canMessageIds();
  std::vector<MessageTableRow> rows;
  rows.reserve(std::min(ids.size(), state.max_rows));
  for (const MessageId &id : ids) {
    if (state.bus_filter >= 0 && id.source != static_cast<uint8_t>(state.bus_filter)) continue;
    if (!message_id_matches_filter(id, state.filter)) continue;
    rows.push_back({id, summarize_message_events(store, id, range)});
    if (rows.size() >= state.max_rows) break;
  }
  return rows;
}

void draw_messages_pane(Session &session, PaneInstance &pane);

}  // namespace loggy
