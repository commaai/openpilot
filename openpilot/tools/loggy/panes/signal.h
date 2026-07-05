#pragma once

#include "tools/loggy/backend/dbc/dbc.h"
#include "tools/loggy/panes/binary.h"

#include "json11/json11.hpp"

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>
#include <vector>

namespace loggy {

struct SignalPaneState {
  std::string filter;
  size_t max_rows = 512;
};

struct SignalPaneRow {
  std::string name;
  std::string kind;
  int start_bit = 0;
  int size = 1;
  std::string endian;
  std::string value;
  uint32_t flip_count = 0;
  bool from_dbc = false;
};

inline SignalPaneState parse_signal_pane_state(std::string_view state_json) {
  SignalPaneState state;
  std::string err;
  const json11::Json json = json11::Json::parse(std::string(state_json), err);
  if (!err.empty() || !json.is_object()) return state;
  if (json["filter"].is_string()) state.filter = json["filter"].string_value();
  if (json["max_rows"].is_number()) state.max_rows = static_cast<size_t>(std::clamp(json["max_rows"].int_value(), 16, 5000));
  return state;
}

inline std::string signal_pane_state_json(const SignalPaneState &state) {
  return json11::Json(json11::Json::object{
    {"filter", state.filter},
    {"max_rows", static_cast<int>(state.max_rows)},
  }).dump();
}

inline std::string signal_lower_text(std::string_view text) {
  std::string out(text);
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}

inline bool signal_text_matches_filter(std::string_view text, std::string_view filter) {
  return filter.empty() || signal_lower_text(text).find(signal_lower_text(filter)) != std::string::npos;
}

inline const char *signal_type_label(Signal::Type type) {
  switch (type) {
    case Signal::Type::Multiplexed: return "Muxed";
    case Signal::Type::Multiplexor: return "Mux";
    case Signal::Type::Normal:
    default: return "Signal";
  }
}

inline std::vector<SignalPaneRow> prepare_signal_pane_rows(const Store &store,
                                                           const MessageId &id,
                                                           TimeRange range,
                                                           const SignalPaneState &state,
                                                           Msg *msg = nullptr) {
  const MessageSummary summary = summarize_message_events(store, id, range);
  std::vector<SignalPaneRow> rows;

  if (msg != nullptr && !msg->getSignals().empty()) {
    rows.reserve(std::min(msg->getSignals().size(), state.max_rows));
    for (const Signal *sig : msg->getSignals()) {
      if (sig == nullptr) continue;
      const std::string searchable = sig->name + " " + sig->unit + " " + sig->comment;
      if (!signal_text_matches_filter(searchable, state.filter)) continue;

      std::string value = "--";
      if (!summary.latest_data.empty()) {
        double decoded = 0.0;
        if (sig->getValue(summary.latest_data.data(), summary.latest_data.size(), &decoded)) {
          value = sig->formatValue(decoded);
        }
      }
      rows.push_back({
        .name = sig->name,
        .kind = signal_type_label(sig->type),
        .start_bit = sig->start_bit,
        .size = sig->size,
        .endian = sig->is_little_endian ? "LE" : "BE",
        .value = std::move(value),
        .flip_count = 0,
        .from_dbc = true,
      });
      if (rows.size() >= state.max_rows) break;
    }
    return rows;
  }

  const std::optional<BinaryGrid> grid = build_binary_grid(store, id, range);
  if (!grid.has_value()) return rows;
  rows.reserve(std::min(grid->rows.size() * 8, state.max_rows));
  for (size_t byte_index = 0; byte_index < grid->rows.size(); ++byte_index) {
    for (int bit_column = 0; bit_column < 8; ++bit_column) {
      const int bit = 7 - bit_column;
      const int start_bit = static_cast<int>(byte_index * 8 + static_cast<size_t>(bit));
      const BinaryBitCell &cell = grid->rows[byte_index][static_cast<size_t>(bit_column)];
      std::string name = "byte" + std::to_string(byte_index) + ".bit" + std::to_string(bit);
      if (!signal_text_matches_filter(name, state.filter)) continue;
      rows.push_back({
        .name = std::move(name),
        .kind = "Bit",
        .start_bit = start_bit,
        .size = 1,
        .endian = "-",
        .value = cell.value ? "1" : "0",
        .flip_count = cell.flip_count,
        .from_dbc = false,
      });
      if (rows.size() >= state.max_rows) return rows;
    }
  }
  return rows;
}

void draw_signal_pane(Session &session, PaneInstance &pane);

}  // namespace loggy
