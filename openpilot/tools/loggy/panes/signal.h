#pragma once

#include "tools/loggy/backend/dbc/dbc.h"
#include "tools/loggy/backend/undo.h"
#include "tools/loggy/panes/binary.h"

#include "json11/json11.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace loggy {

struct SignalPaneState {
  std::string filter;
  std::string selected_signal;
  std::string edit_error;
  size_t max_rows = 512;
  int sparkline_seconds = 30;
};

struct SignalSparkline {
  std::vector<double> values;
  double min = 0.0;
  double max = 0.0;
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
  const Signal *signal = nullptr;
  SignalSparkline sparkline;
};

struct SignalEditModel {
  std::string original_name;
  std::string name;
  int start_bit = 0;
  int size = 1;
  bool is_little_endian = true;
  bool is_signed = false;
  double factor = 1.0;
  double offset = 0.0;
  double min = 0.0;
  double max = 1.0;
  std::string unit;
  std::string receiver;
  std::string comment;
  ValueDescription val_desc;
  Signal::Type type = Signal::Type::Normal;
  int multiplex_value = 0;
};

inline SignalPaneState parse_signal_pane_state(std::string_view state_json) {
  SignalPaneState state;
  std::string err;
  const json11::Json json = json11::Json::parse(std::string(state_json), err);
  if (!err.empty() || !json.is_object()) return state;
  if (json["filter"].is_string()) state.filter = json["filter"].string_value();
  if (json["selected_signal"].is_string()) state.selected_signal = json["selected_signal"].string_value();
  if (json["edit_error"].is_string()) state.edit_error = json["edit_error"].string_value();
  if (json["max_rows"].is_number()) state.max_rows = static_cast<size_t>(std::clamp(json["max_rows"].int_value(), 16, 5000));
  if (json["sparkline_seconds"].is_number()) state.sparkline_seconds = std::clamp(json["sparkline_seconds"].int_value(), 1, 120);
  return state;
}

inline std::string signal_pane_state_json(const SignalPaneState &state) {
  return json11::Json(json11::Json::object{
    {"filter", state.filter},
    {"selected_signal", state.selected_signal},
    {"edit_error", state.edit_error},
    {"max_rows", static_cast<int>(state.max_rows)},
    {"sparkline_seconds", state.sparkline_seconds},
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

inline std::string signal_value_descriptions_text(const ValueDescription &descriptions) {
  std::string out;
  for (const auto &[value, description] : descriptions) {
    if (!out.empty()) out += ' ';
    out += doubleToString(value) + " \"";
    for (char ch : description) {
      if (ch == '"' || ch == '\\') out += '\\';
      out += ch;
    }
    out += '"';
  }
  return out;
}

inline bool parse_signal_value_descriptions(std::string_view text, ValueDescription *descriptions,
                                            std::string *error = nullptr) {
  ValueDescription parsed;
  size_t pos = 0;
  const auto skip_spaces = [&]() {
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) ++pos;
  };

  skip_spaces();
  while (pos < text.size()) {
    const std::string value_text(text.substr(pos));
    char *end = nullptr;
    const double value = std::strtod(value_text.c_str(), &end);
    if (end == value_text.c_str() || !std::isfinite(value)) {
      if (error != nullptr) *error = "value description entry needs a numeric value";
      return false;
    }
    pos += static_cast<size_t>(end - value_text.c_str());
    skip_spaces();
    if (pos >= text.size() || text[pos] != '"') {
      if (error != nullptr) *error = "value description entry needs a quoted description";
      return false;
    }
    ++pos;

    std::string description;
    bool closed = false;
    while (pos < text.size()) {
      const char ch = text[pos++];
      if (ch == '\\' && pos < text.size()) {
        description += text[pos++];
      } else if (ch == '"') {
        closed = true;
        break;
      } else {
        description += ch;
      }
    }
    if (!closed) {
      if (error != nullptr) *error = "value description is missing a closing quote";
      return false;
    }
    parsed.push_back({value, std::move(description)});
    skip_spaces();
  }

  if (descriptions != nullptr) *descriptions = std::move(parsed);
  if (error != nullptr) error->clear();
  return true;
}

inline SignalSparkline prepare_signal_sparkline(const Store &store, const MessageId &id,
                                                TimeRange range, const Signal &signal,
                                                size_t max_points = 48,
                                                double window_seconds = 0.0) {
  SignalSparkline sparkline;
  if (max_points == 0) return sparkline;

  const CanEventView view = store.canEvents(id, range);
  const double min_time = (window_seconds > 0.0 && !view.events.empty())
                            ? view.events.back().mono_time - window_seconds
                            : -std::numeric_limits<double>::infinity();
  sparkline.values.reserve(std::min(view.events.size(), max_points));
  const size_t step = view.events.size() <= max_points ? 1 : (view.events.size() + max_points - 1) / max_points;
  double min_value = std::numeric_limits<double>::infinity();
  double max_value = -std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < view.events.size(); i += step) {
    const CanEvent &event = view.events[i];
    if (event.mono_time < min_time) continue;
    double value = 0.0;
    if (!signal.getValue(event.data.data(), event.data.size(), &value)) continue;
    sparkline.values.push_back(value);
    min_value = std::min(min_value, value);
    max_value = std::max(max_value, value);
  }
  if (!sparkline.values.empty()) {
    sparkline.min = min_value;
    sparkline.max = max_value;
  }
  return sparkline;
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
      const std::string searchable = sig->name + " " + sig->unit + " " + sig->comment + " " +
                                     signal_value_descriptions_text(sig->val_desc);
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
        .signal = sig,
        .sparkline = prepare_signal_sparkline(store, id, range, *sig, 48, static_cast<double>(state.sparkline_seconds)),
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
        .signal = nullptr,
        .sparkline = {},
      });
      if (rows.size() >= state.max_rows) return rows;
    }
  }
  return rows;
}

inline SignalEditModel signal_edit_model_from_signal(const Signal &signal) {
  return {
    .original_name = signal.name,
    .name = signal.name,
    .start_bit = signal.start_bit,
    .size = signal.size,
    .is_little_endian = signal.is_little_endian,
    .is_signed = signal.is_signed,
    .factor = signal.factor,
    .offset = signal.offset,
    .min = signal.min,
    .max = signal.max,
    .unit = signal.unit,
    .receiver = signal.receiver_name,
    .comment = signal.comment,
    .val_desc = signal.val_desc,
    .type = signal.type,
    .multiplex_value = signal.multiplex_value,
  };
}

inline Signal signal_from_edit_model(const SignalEditModel &edit, const Signal &origin) {
  Signal signal = origin;
  signal.name = edit.name;
  signal.start_bit = edit.start_bit;
  signal.size = edit.size;
  signal.is_little_endian = edit.is_little_endian;
  signal.is_signed = edit.is_signed;
  signal.factor = edit.factor;
  signal.offset = edit.offset;
  signal.min = edit.min;
  signal.max = edit.max;
  signal.unit = edit.unit;
  signal.receiver_name = edit.receiver;
  signal.comment = edit.comment;
  signal.val_desc = edit.val_desc;
  signal.type = edit.type;
  signal.multiplex_value = edit.multiplex_value;
  signal.update();
  return signal;
}

inline bool signal_edit_model_changed(const SignalEditModel &edit, const Signal &origin) {
  const Signal edited = signal_from_edit_model(edit, origin);
  return edited != origin;
}

inline bool apply_signal_edit(UndoStack &undo_stack, DBCManager &manager, const MessageId &id,
                              const SignalEditModel &edit, std::string *error = nullptr) {
  Msg *msg = manager.msg(id);
  if (msg == nullptr) {
    if (error != nullptr) *error = "no DBC message for " + id.toString();
    return false;
  }
  const Signal *origin = msg->sig(edit.original_name);
  if (origin == nullptr) {
    if (error != nullptr) *error = "signal no longer exists: " + edit.original_name;
    return false;
  }
  return commit_signal_edit(&undo_stack, &manager, id, origin, signal_from_edit_model(edit, *origin), error);
}

inline bool remove_signal_edit(UndoStack &undo_stack, DBCManager &manager, const MessageId &id,
                               const std::string &signal_name, std::string *error = nullptr) {
  if (signal_name.empty()) {
    if (error != nullptr) *error = "no DBC signal selected";
    return false;
  }
  return commit_signal_remove(&undo_stack, &manager, id, signal_name, error);
}

void draw_signal_pane(Session &session, PaneInstance &pane);

}  // namespace loggy
