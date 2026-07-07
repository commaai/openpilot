#include "tools/loggy/panes/signal.h"

#include "tools/loggy/backend/csv.h"
#include "tools/loggy/backend/dbc/dbcmanager.h"
#include "tools/loggy/backend/dbc/undo.h"
#include "tools/loggy/backend/session.h"
#include "tools/loggy/shell/theme.h"
#include "tools/loggy/shell/workspace.h"

#include "imgui.h"
#include "json11/json11.hpp"

#include <algorithm>
#include <array>
#include <any>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loggy {
namespace {

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
  // Seconds since the covering byte last changed (within the fade window ending at the
  // tracker); < 0 means no recent change. Bit-candidate rows only -- DBC rows show "--".
  double byte_change_age = -1.0;
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
  int precision = 0;
  ColorRGBA color;
  bool precision_override = false;
  bool color_override = false;
  Signal::Type type = Signal::Type::Normal;
  int multiplex_value = 0;
};

struct MessageEditModel {
  std::string name;
  int size = 8;
  std::string transmitter;
  std::string comment;
};

struct SignalEditCache {
  MessageId id;
  SignalEditModel edit;
  std::string val_desc_text;
  std::string val_desc_error;
  bool val_desc_valid = true;
  bool valid = false;
  // DBC mutation generation the buffer was loaded at: an undo/redo/apply from ANYWHERE (global
  // Ctrl+Z, the DBC tab) must reload the staged text, or the editor shows the pre-undo values
  // with Apply armed to silently re-apply them.
  uint64_t dbc_generation = 0;
};

struct MessageEditCache {
  MessageId id;
  MessageEditModel edit;
  bool valid = false;
  uint64_t dbc_generation = 0;  // same contract as SignalEditCache
};

struct SignalPaneTransientState {
  SignalPaneState state;
  std::string loaded_json;
  SignalEditCache edit;
  MessageEditCache message;
  // Draft edits for signals other than the one currently shown, keyed by (message, original
  // signal name), so switching the table selection away and back doesn't discard unsaved work.
  // Bounded by clearing whenever the active DBC file set changes (dbc_generation, below) — a
  // stale message id or signal name can otherwise never be evicted since selection alone never
  // shrinks the map.
  std::map<std::pair<MessageId, std::string>, SignalEditModel> pending_signal_edits;
  int dbc_generation = -1;
};

bool signal_color_equal(ColorRGBA a, ColorRGBA b) {
  return a.r == b.r && a.g == b.g && a.b == b.b && a.a == b.a;
}

SignalPaneState parse_signal_pane_state(std::string_view state_json) {
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

std::string signal_pane_state_json(const SignalPaneState &state) {
  return json11::Json(json11::Json::object{
    {"filter", state.filter},
    {"selected_signal", state.selected_signal},
    {"edit_error", state.edit_error},
    {"max_rows", static_cast<int>(state.max_rows)},
    {"sparkline_seconds", state.sparkline_seconds},
  }).dump();
}

std::string signal_lower_text(std::string_view text) {
  std::string out(text);
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}

bool signal_text_matches_filter(std::string_view text, std::string_view filter) {
  return filter.empty() || signal_lower_text(text).find(signal_lower_text(filter)) != std::string::npos;
}

const char *signal_type_label(Signal::Type type) {
  switch (type) {
    case Signal::Type::Multiplexed: return "Muxed";
    case Signal::Type::Multiplexor: return "Mux";
    case Signal::Type::Normal:
    default: return "Signal";
  }
}

std::string signal_value_descriptions_text(const ValueDescription &descriptions) {
  std::string out;
  for (const auto &[value, description] : descriptions) {
    if (!out.empty()) out += ' ';
    out += double_to_string(value) + " \"";
    for (char ch : description) {
      if (ch == '"' || ch == '\\') out += '\\';
      out += ch;
    }
    out += '"';
  }
  return out;
}

std::optional<ValueDescription> parse_signal_value_descriptions(std::string_view text, std::string &error) {
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
      error = "value description entry needs a numeric value";
      return std::nullopt;
    }
    pos += static_cast<size_t>(end - value_text.c_str());
    skip_spaces();
    if (pos >= text.size() || text[pos] != '"') {
      error = "value description entry needs a quoted description";
      return std::nullopt;
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
      error = "value description is missing a closing quote";
      return std::nullopt;
    }
    parsed.push_back({value, std::move(description)});
    skip_spaces();
  }

  error.clear();
  return parsed;
}

// `window` anchors the sparkline at the tracker -- [tracker - sparkline_seconds, tracker] -- not
// at the chart's zoom/view range, so it neither freezes when the tracker scrolls off a zoomed
// view nor jumps when the user zooms (REVIEW.md's playhead-semantics cluster).
SignalSparkline prepare_signal_sparkline(const Store &store, const MessageId &id, TimeRange window,
                                        const Signal &signal, size_t max_points = 48) {
  SignalSparkline sparkline;
  if (max_points == 0) return sparkline;

  const CanEventView view = store.can_events(id, window);
  sparkline.values.reserve(std::min(view.events.size(), max_points));
  const size_t step = view.events.size() <= max_points ? 1 : (view.events.size() + max_points - 1) / max_points;
  double min_value = std::numeric_limits<double>::infinity();
  double max_value = -std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < view.events.size(); i += step) {
    const CanEvent &event = view.events[i];
    double value = 0.0;
    if (!signal.get_value(event.data.data(), event.data.size(), &value)) continue;
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

// `state_range` is {route_start, tracker}: Value/bit-candidates reflect the newest event <=
// tracker, like Binary/Messages. `sparkline_range` is [tracker - sparkline_seconds, tracker],
// independent of both `state_range` and any chart zoom.
std::vector<SignalPaneRow> prepare_signal_pane_rows(const Store &store, const MessageId &id, TimeRange state_range,
                                                   TimeRange sparkline_range, const SignalPaneState &state,
                                                   Msg *msg = nullptr) {
  const MessageSummary summary = summarize_message_events(store, id, state_range);
  std::vector<SignalPaneRow> rows;

  if (msg != nullptr && !msg->signals().empty()) {
    rows.reserve(std::min(msg->signals().size(), state.max_rows));
    for (const Signal *sig : msg->signals()) {
      if (sig == nullptr) continue;
      const std::string searchable = sig->name + " " + sig->unit + " " + sig->comment + " " +
                                     signal_value_descriptions_text(sig->val_desc);
      if (!signal_text_matches_filter(searchable, state.filter)) continue;

      std::string value = "--";
      if (!summary.latest_data.empty()) {
        double decoded = 0.0;
        if (sig->get_value(summary.latest_data.data(), summary.latest_data.size(), &decoded)) {
          value = sig->format_value(decoded);
        }
      }
      rows.push_back({
        .name = sig->name,
        .kind = signal_type_label(sig->type),
        .start_bit = sig->start_bit,
        .size = sig->size,
        .endian = sig->is_little_endian ? "LE" : "BE",
        .value = std::move(value),
        .byte_change_age = -1.0,
        .from_dbc = true,
        .signal = sig,
        .sparkline = prepare_signal_sparkline(store, id, sparkline_range, *sig, 48),
      });
      if (rows.size() >= state.max_rows) break;
    }
    return rows;
  }

  const std::optional<BinaryGrid> grid = build_binary_grid(store, id, state_range);
  if (!grid.has_value()) return rows;
  rows.reserve(std::min(grid->rows.size() * 8, state.max_rows));
  for (size_t byte_index = 0; byte_index < grid->rows.size(); ++byte_index) {
    const double last_change = byte_index < grid->byte_last_change.size()
                                  ? grid->byte_last_change[byte_index]
                                  : -std::numeric_limits<double>::infinity();
    const double byte_change_age = std::isfinite(last_change) ? (state_range.end - last_change) : -1.0;
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
        .byte_change_age = byte_change_age,
        .from_dbc = false,
        .signal = nullptr,
        .sparkline = {},
      });
      if (rows.size() >= state.max_rows) return rows;
    }
  }
  return rows;
}

SignalEditModel signal_edit_model_from_signal(const Signal &signal) {
  Signal automatic = signal;
  automatic.precision_override = false;
  automatic.color_override = false;
  automatic.update();

  SignalEditModel edit;
  edit.original_name = signal.name;
  edit.name = signal.name;
  edit.start_bit = signal.start_bit;
  edit.size = signal.size;
  edit.is_little_endian = signal.is_little_endian;
  edit.is_signed = signal.is_signed;
  edit.factor = signal.factor;
  edit.offset = signal.offset;
  edit.min = signal.min;
  edit.max = signal.max;
  edit.unit = signal.unit;
  edit.receiver = signal.receiver_name;
  edit.comment = signal.comment;
  edit.val_desc = signal.val_desc;
  edit.precision = signal.precision;
  edit.color = signal.color;
  edit.precision_override = signal.precision_override || signal.precision != automatic.precision;
  edit.color_override = signal.color_override || !signal_color_equal(signal.color, automatic.color);
  edit.type = signal.type;
  edit.multiplex_value = signal.multiplex_value;
  return edit;
}

Signal signal_from_edit_model(const SignalEditModel &edit, const Signal &origin, bool apply_overrides = true) {
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
  signal.precision_override = false;
  signal.color_override = false;
  signal.update();
  if (apply_overrides && edit.precision_override) {
    signal.precision = std::clamp(edit.precision, 0, 12);
    signal.precision_override = true;
  }
  if (apply_overrides && edit.color_override) {
    signal.color = edit.color;
    signal.color_override = true;
  }
  return signal;
}

Signal automatic_signal_from_edit_model(const SignalEditModel &edit, const Signal &origin) {
  return signal_from_edit_model(edit, origin, false);
}

bool signal_edit_model_changed(const SignalEditModel &edit, const Signal &origin) {
  const Signal edited = signal_from_edit_model(edit, origin);
  return edited != origin;
}

bool apply_signal_edit(UndoStack &undo_stack, DBCManager &manager, const MessageId &id, const SignalEditModel &edit,
                      std::string &error) {
  Msg *msg = manager.msg(id);
  if (msg == nullptr) {
    error = "no DBC message for " + id.to_string();
    return false;
  }
  const Signal *origin = msg->sig(edit.original_name);
  if (origin == nullptr) {
    error = "signal no longer exists: " + edit.original_name;
    return false;
  }
  return commit_signal_edit(undo_stack, manager, id, *origin, signal_from_edit_model(edit, *origin), error);
}

bool remove_signal_edit(UndoStack &undo_stack, DBCManager &manager, const MessageId &id, const std::string &signal_name,
                       std::string &error) {
  if (signal_name.empty()) {
    error = "no DBC signal selected";
    return false;
  }
  return commit_signal_remove(undo_stack, manager, id, signal_name, error);
}

MessageEditModel message_edit_model_from_msg(const Msg &msg) {
  MessageEditModel edit;
  edit.name = msg.name;
  edit.size = static_cast<int>(msg.size);
  edit.transmitter = msg.transmitter;
  edit.comment = msg.comment;
  return edit;
}

Msg message_from_edit_model(const MessageEditModel &edit, const Msg &origin) {
  Msg msg = origin;
  msg.name = edit.name;
  msg.size = edit.size < 0 ? 0U : static_cast<uint32_t>(edit.size);
  msg.transmitter = edit.transmitter.empty() ? DEFAULT_NODE_NAME : edit.transmitter;
  msg.comment = edit.comment;
  if (msg.size > 0 && msg.size <= CAN_MAX_DATA_BYTES) msg.update();
  return msg;
}

bool message_edit_model_changed(const MessageEditModel &edit, const Msg &origin) {
  const Msg edited = message_from_edit_model(edit, origin);
  return edited.name != origin.name ||
         edited.size != origin.size ||
         edited.transmitter != origin.transmitter ||
         edited.comment != origin.comment;
}

bool apply_message_edit(UndoStack &undo_stack, DBCManager &manager, const MessageId &id, const MessageEditModel &edit,
                       std::string &error) {
  Msg *origin = manager.msg(id);
  if (origin == nullptr) {
    error = "no DBC message for " + id.to_string();
    return false;
  }
  return commit_message_edit(undo_stack, manager, id, message_from_edit_model(edit, *origin), error);
}

SignalPaneTransientState &signal_pane_transient_state(PaneInstance &pane) {
  if (SignalPaneTransientState *state = std::any_cast<SignalPaneTransientState>(&pane.transient_state)) {
    return *state;
  }
  pane.transient_state = SignalPaneTransientState{};
  return std::any_cast<SignalPaneTransientState &>(pane.transient_state);
}

SignalPaneState &signal_pane_state(PaneInstance &pane, SignalPaneTransientState &transient) {
  if (transient.loaded_json != pane.state_json) {
    transient.state = parse_signal_pane_state(pane.state_json);
    transient.loaded_json = pane.state_json;
  }
  return transient.state;
}

void save_signal_pane_state(PaneInstance &pane, SignalPaneTransientState &transient) {
  pane.state_json = signal_pane_state_json(transient.state);
  transient.loaded_json = pane.state_json;
}

int signal_type_combo_index(Signal::Type type) {
  switch (type) {
    case Signal::Type::Multiplexed: return 1;
    case Signal::Type::Multiplexor: return 2;
    case Signal::Type::Normal:
    default: return 0;
  }
}

Signal::Type signal_type_from_combo_index(int index) {
  switch (index) {
    case 1: return Signal::Type::Multiplexed;
    case 2: return Signal::Type::Multiplexor;
    case 0:
    default: return Signal::Type::Normal;
  }
}

void load_signal_edit_cache(SignalEditCache *cache, const MessageId &id, const Signal &signal,
                            uint64_t dbc_generation) {
  if (cache == nullptr) return;
  cache->dbc_generation = dbc_generation;
  cache->id = id;
  cache->edit = signal_edit_model_from_signal(signal);
  cache->val_desc_text = signal_value_descriptions_text(signal.val_desc);
  cache->val_desc_error.clear();
  cache->val_desc_valid = true;
  cache->valid = true;
}

bool cache_matches_signal(const SignalEditCache &cache, const MessageId &id, const Signal &signal,
                          uint64_t dbc_generation) {
  return cache.valid && cache.dbc_generation == dbc_generation && cache.id == id &&
         cache.edit.original_name == signal.name;
}

bool signal_edit_cache_dirty(const SignalEditCache &cache, const Signal &signal) {
  return cache.valid && cache.val_desc_valid && signal_edit_model_changed(cache.edit, signal);
}

// Called before switching the selected signal away from whatever the cache currently holds. If
// that signal still exists and the cache has unsaved changes for it, stash the edit model so
// reselecting the same signal restores it instead of silently reloading from the DBC.
void stash_pending_signal_edit(SignalPaneTransientState &transient, const MessageId &id, Msg *msg) {
  const SignalEditCache &cache = transient.edit;
  if (!cache.valid || msg == nullptr) return;
  const Signal *origin = msg->sig(cache.edit.original_name);
  if (origin == nullptr || !signal_edit_cache_dirty(cache, *origin)) return;
  transient.pending_signal_edits[{id, cache.edit.original_name}] = cache.edit;
}

// Called after switching to a new signal. If a stash exists for it (keyed by its current name,
// which is also its original_name for any signal not itself mid-rename), load it into the cache
// and report success so the caller skips the normal fresh-from-DBC reload.
bool restore_pending_signal_edit(SignalPaneTransientState &transient, const MessageId &id, const Signal &signal,
                                 uint64_t dbc_generation) {
  const auto it = transient.pending_signal_edits.find({id, signal.name});
  if (it == transient.pending_signal_edits.end()) return false;
  SignalEditCache &cache = transient.edit;
  cache.dbc_generation = dbc_generation;
  cache.id = id;
  cache.edit = it->second;
  cache.val_desc_text = signal_value_descriptions_text(cache.edit.val_desc);
  cache.val_desc_error.clear();
  cache.val_desc_valid = true;
  cache.valid = true;
  return true;
}

void discard_pending_signal_edit(SignalPaneTransientState &transient, const MessageId &id, const std::string &original_name) {
  transient.pending_signal_edits.erase({id, original_name});
}

void load_message_edit_cache(MessageEditCache *cache, const MessageId &id, const Msg &msg,
                             uint64_t dbc_generation) {
  if (cache == nullptr) return;
  cache->dbc_generation = dbc_generation;
  cache->id = id;
  cache->edit = message_edit_model_from_msg(msg);
  cache->valid = true;
}

bool cache_matches_message(const MessageEditCache &cache, const MessageId &id, uint64_t dbc_generation) {
  return cache.valid && cache.dbc_generation == dbc_generation && cache.id == id;
}

uint8_t color_channel_from_float(float value) {
  return static_cast<uint8_t>(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
}

void draw_message_editor(Session &session, const MessageId &id, SignalPaneState *state, MessageEditCache *cache,
                        SignalEditCache *signal_cache, bool *changed) {
  if (state == nullptr || cache == nullptr || changed == nullptr) return;
  Msg *msg = session.dbc.msg(id);
  if (msg == nullptr) {
    cache->valid = false;
    return;
  }
  if (!cache_matches_message(*cache, id, session.dbc.generation())) {
    load_message_edit_cache(cache, id, *msg, session.dbc.generation());
  }

  MessageEditModel &edit = cache->edit;
  ImGui::Separator();
  // Collapsed by default: the signal LIST is the primary content (like cabana), so the message
  // header/rename/resize editor stays out of the way until the user opens it. Otherwise the
  // always-expanded editor starved the signal table to zero rows in the cabana preset.
  if (!ImGui::CollapsingHeader("Message")) return;

  ImGui::SetNextItemWidth(180.0f);
  if (input_text_with_hint("Name##message", "", &edit.name)) *changed = true;
  if (ImGui::GetContentRegionAvail().x > 108.0f) ImGui::SameLine();
  ImGui::SetNextItemWidth(78.0f);
  if (ImGui::InputInt("Bytes", &edit.size)) *changed = true;
  if (ImGui::GetContentRegionAvail().x > 156.0f) ImGui::SameLine();
  ImGui::SetNextItemWidth(136.0f);
  if (input_text_with_hint("Transmitter", "", &edit.transmitter)) *changed = true;

  ImGui::SetNextItemWidth(std::clamp(ImGui::GetContentRegionAvail().x * 0.55f, 220.0f, 520.0f));
  if (input_text_with_hint("Comment##message", "", &edit.comment)) *changed = true;

  const bool has_edit = message_edit_model_changed(edit, *msg);
  if (!has_edit) ImGui::BeginDisabled();
  if (ImGui::Button("Apply##message")) {
    std::string error;
    if (apply_message_edit(session.dbc_undo, session.dbc, id, edit, error)) {
      state->edit_error.clear();
      cache->valid = false;
      if (signal_cache != nullptr) signal_cache->valid = false;
    } else {
      state->edit_error = error;
    }
    *changed = true;
  }
  if (!has_edit) ImGui::EndDisabled();

  if (ImGui::GetContentRegionAvail().x > 72.0f) ImGui::SameLine();
  if (ImGui::Button("Reset##message")) {
    load_message_edit_cache(cache, id, *msg, session.dbc.generation());
    state->edit_error.clear();
    *changed = true;
  }

  if (ImGui::GetContentRegionAvail().x > 96.0f) ImGui::SameLine();
  UndoStack &undo = session.dbc_undo;
  const bool undo_disabled = !undo.can_undo();
  if (undo_disabled) ImGui::BeginDisabled();
  if (ImGui::Button("Undo##message")) {
    undo.undo();
    cache->valid = false;
    if (signal_cache != nullptr) signal_cache->valid = false;
    state->edit_error.clear();
    *changed = true;
  }
  if (undo_disabled) ImGui::EndDisabled();
  if (undo.can_undo() && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", undo.undo_text().c_str());

  if (ImGui::GetContentRegionAvail().x > 96.0f) ImGui::SameLine();
  const bool redo_disabled = !undo.can_redo();
  if (redo_disabled) ImGui::BeginDisabled();
  if (ImGui::Button("Redo##message")) {
    undo.redo();
    cache->valid = false;
    if (signal_cache != nullptr) signal_cache->valid = false;
    state->edit_error.clear();
    *changed = true;
  }
  if (redo_disabled) ImGui::EndDisabled();
  if (undo.can_redo() && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", undo.redo_text().c_str());

  if (!state->edit_error.empty()) {
    ImGui::SameLine();
    ImGui::TextDisabled("%s", state->edit_error.c_str());
  }
}

void draw_signal_sparkline(const SignalPaneRow &row) {
  constexpr float width = 92.0f;
  const float height = std::max(18.0f, ImGui::GetTextLineHeight() + 4.0f);
  const ImVec2 pos = ImGui::GetCursorScreenPos();
  ImGui::Dummy(ImVec2(width, height));

  ImDrawList *draw_list = ImGui::GetWindowDrawList();
  const ImVec2 max(pos.x + width, pos.y + height);
  draw_list->AddRectFilled(pos, max, ImGui::GetColorU32(theme().sparkline_bg), 2.0f);
  draw_list->AddRect(pos, max, ImGui::GetColorU32(theme().sparkline_border), 2.0f);

  if (row.sparkline.values.empty()) {
    draw_list->AddText(ImVec2(pos.x + 4.0f, pos.y + 2.0f), ImGui::GetColorU32(ImGuiCol_TextDisabled), "--");
    return;
  }

  const ColorRGBA color = row.signal == nullptr ? ColorRGBA{180, 180, 180, 255} : row.signal->color;
  const ImU32 line_color = IM_COL32(color.r, color.g, color.b, 255);
  const double raw_span = row.sparkline.max - row.sparkline.min;
  const double span = std::max(raw_span, 1e-9);
  std::vector<ImVec2> points;
  points.reserve(row.sparkline.values.size());
  for (size_t i = 0; i < row.sparkline.values.size(); ++i) {
    const float x = pos.x + 2.0f + (width - 4.0f) * (row.sparkline.values.size() == 1 ? 0.5f : static_cast<float>(i) / static_cast<float>(row.sparkline.values.size() - 1));
    const double normalized = raw_span <= 1e-9 ? 0.5 : (row.sparkline.max - row.sparkline.values[i]) / span;
    const float y = pos.y + 2.0f + (height - 4.0f) * static_cast<float>(normalized);
    points.push_back(ImVec2(x, y));
  }
  if (points.size() == 1) {
    draw_list->AddCircleFilled(points.front(), 2.0f, line_color);
  } else {
    draw_list->AddPolyline(points.data(), static_cast<int>(points.size()), line_color, 0, 1.5f);
  }
}

void draw_signal_editor(Session &session, const MessageId &id, SignalPaneState *state, SignalEditCache *cache,
                       MessageEditCache *message_cache, SignalPaneTransientState *transient, bool *changed) {
  if (state == nullptr || cache == nullptr || changed == nullptr || state->selected_signal.empty()) return;

  Msg *msg = session.dbc.msg(id);
  Signal *signal = msg == nullptr ? nullptr : msg->sig(state->selected_signal);
  if (signal == nullptr) {
    state->selected_signal.clear();
    cache->valid = false;
    *changed = true;
    return;
  }
  if (!cache_matches_signal(*cache, id, *signal, session.dbc.generation())) {
    load_signal_edit_cache(cache, id, *signal, session.dbc.generation());
  }

  SignalEditModel &edit = cache->edit;
  const bool dirty = signal_edit_cache_dirty(*cache, *signal);
  ImGui::Separator();
  push_bold_font();
  ImGui::TextUnformatted("Signal");
  pop_bold_font();
  if (dirty) {
    ImGui::SameLine();
    ImGui::TextDisabled("(unapplied edits)");
  }

  ImGui::SetNextItemWidth(180.0f);
  if (input_text_with_hint("Name", "", &edit.name)) *changed = true;
  if (ImGui::GetContentRegionAvail().x > 120.0f) ImGui::SameLine();
  ImGui::SetNextItemWidth(86.0f);
  if (ImGui::InputInt("Start", &edit.start_bit)) *changed = true;
  if (ImGui::GetContentRegionAvail().x > 112.0f) ImGui::SameLine();
  ImGui::SetNextItemWidth(76.0f);
  if (ImGui::InputInt("Size", &edit.size)) *changed = true;

  if (ImGui::Checkbox("Little Endian", &edit.is_little_endian)) *changed = true;
  if (ImGui::GetContentRegionAvail().x > 96.0f) ImGui::SameLine();
  if (ImGui::Checkbox("Signed", &edit.is_signed)) *changed = true;
  if (ImGui::GetContentRegionAvail().x > 128.0f) ImGui::SameLine();
  int type_index = signal_type_combo_index(edit.type);
  ImGui::SetNextItemWidth(112.0f);
  if (ImGui::Combo("Type", &type_index, "Normal\0Muxed\0Mux\0")) {
    edit.type = signal_type_from_combo_index(type_index);
    *changed = true;
  }
  if (edit.type == Signal::Type::Multiplexed) {
    if (ImGui::GetContentRegionAvail().x > 116.0f) ImGui::SameLine();
    ImGui::SetNextItemWidth(86.0f);
    if (ImGui::InputInt("Mux", &edit.multiplex_value)) *changed = true;
  }

  ImGui::SetNextItemWidth(108.0f);
  if (ImGui::InputDouble("Factor", &edit.factor, 0.0, 0.0, "%.9g")) *changed = true;
  if (ImGui::GetContentRegionAvail().x > 124.0f) ImGui::SameLine();
  ImGui::SetNextItemWidth(108.0f);
  if (ImGui::InputDouble("Offset", &edit.offset, 0.0, 0.0, "%.9g")) *changed = true;
  if (ImGui::GetContentRegionAvail().x > 112.0f) ImGui::SameLine();
  ImGui::SetNextItemWidth(108.0f);
  if (ImGui::InputDouble("Min", &edit.min, 0.0, 0.0, "%.9g")) *changed = true;
  if (ImGui::GetContentRegionAvail().x > 112.0f) ImGui::SameLine();
  ImGui::SetNextItemWidth(108.0f);
  if (ImGui::InputDouble("Max", &edit.max, 0.0, 0.0, "%.9g")) *changed = true;

  ImGui::SetNextItemWidth(120.0f);
  if (input_text_with_hint("Unit", "", &edit.unit)) *changed = true;
  if (ImGui::GetContentRegionAvail().x > 164.0f) ImGui::SameLine();
  ImGui::SetNextItemWidth(160.0f);
  if (input_text_with_hint("Receiver", "", &edit.receiver)) *changed = true;
  ImGui::SetNextItemWidth(std::clamp(ImGui::GetContentRegionAvail().x * 0.55f, 220.0f, 520.0f));
  if (input_text_with_hint("Comment", "", &edit.comment)) *changed = true;

  ImGui::SetNextItemWidth(std::clamp(ImGui::GetContentRegionAvail().x * 0.55f, 220.0f, 620.0f));
  if (input_text_with_hint("Value Table", "", &cache->val_desc_text)) {
    std::string parse_error;
    std::optional<ValueDescription> parsed = parse_signal_value_descriptions(cache->val_desc_text, parse_error);
    if (parsed.has_value()) {
      edit.val_desc = std::move(*parsed);
      cache->val_desc_error.clear();
      cache->val_desc_valid = true;
      state->edit_error.clear();
    } else {
      cache->val_desc_error = parse_error;
      cache->val_desc_valid = false;
    }
    *changed = true;
  }
  if (!cache->val_desc_valid) {
    ImGui::SameLine();
    ImGui::TextDisabled("%s", cache->val_desc_error.c_str());
  }

  const Signal automatic = automatic_signal_from_edit_model(edit, *signal);
  if (!edit.precision_override) edit.precision = automatic.precision;
  if (!edit.color_override) edit.color = automatic.color;

  ImGui::SetNextItemWidth(82.0f);
  int precision = edit.precision;
  if (ImGui::InputInt("Precision", &precision)) {
    edit.precision = std::clamp(precision, 0, 12);
    edit.precision_override = true;
    *changed = true;
  }
  if (ImGui::GetContentRegionAvail().x > 96.0f) ImGui::SameLine();
  if (!edit.precision_override) ImGui::BeginDisabled();
  if (ImGui::Button("Auto##signal_precision")) {
    edit.precision = automatic.precision;
    edit.precision_override = false;
    *changed = true;
  }
  if (!edit.precision_override) ImGui::EndDisabled();
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Use precision derived from factor and offset");
  if (!edit.precision_override) {
    ImGui::SameLine();
    ImGui::TextDisabled("auto");
  }

  float color[3] = {
    static_cast<float>(edit.color.r) / 255.0f,
    static_cast<float>(edit.color.g) / 255.0f,
    static_cast<float>(edit.color.b) / 255.0f,
  };
  ImGui::SetNextItemWidth(198.0f);
  if (ImGui::ColorEdit3("Color", color, ImGuiColorEditFlags_NoAlpha | ImGuiColorEditFlags_DisplayRGB)) {
    edit.color = ColorRGBA{
      .r = color_channel_from_float(color[0]),
      .g = color_channel_from_float(color[1]),
      .b = color_channel_from_float(color[2]),
      .a = 255,
    };
    edit.color_override = true;
    *changed = true;
  }
  if (ImGui::GetContentRegionAvail().x > 96.0f) ImGui::SameLine();
  if (!edit.color_override) ImGui::BeginDisabled();
  if (ImGui::Button("Auto##signal_color")) {
    edit.color = automatic.color;
    edit.color_override = false;
    *changed = true;
  }
  if (!edit.color_override) ImGui::EndDisabled();
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Use color derived from signal name and bit position");

  const bool has_edit = cache->val_desc_valid && signal_edit_model_changed(edit, *signal);
  if (!has_edit) ImGui::BeginDisabled();
  if (ImGui::Button(has_edit ? "Apply *##signal_apply" : "Apply##signal_apply")) {
    std::string error;
    if (!cache->val_desc_valid) {
      state->edit_error = cache->val_desc_error;
    } else if (apply_signal_edit(session.dbc_undo, session.dbc, id, edit, error)) {
      if (transient != nullptr) {
        discard_pending_signal_edit(*transient, id, edit.original_name);
      }
      state->selected_signal = edit.name;
      state->edit_error.clear();
      cache->valid = false;
    } else {
      state->edit_error = error;
    }
    *changed = true;
  }
  if (!has_edit) ImGui::EndDisabled();

  if (ImGui::GetContentRegionAvail().x > 72.0f) ImGui::SameLine();
  if (ImGui::Button("Reset")) {
    if (transient != nullptr) {
      discard_pending_signal_edit(*transient, id, edit.original_name);
    }
    load_signal_edit_cache(cache, id, *signal, session.dbc.generation());
    state->edit_error.clear();
    *changed = true;
  }

  if (ImGui::GetContentRegionAvail().x > 80.0f) ImGui::SameLine();
  if (ImGui::Button("Remove")) {
    std::string error;
    if (remove_signal_edit(session.dbc_undo, session.dbc, id, state->selected_signal, error)) {
      if (transient != nullptr) {
        discard_pending_signal_edit(*transient, id, edit.original_name);
      }
      state->selected_signal.clear();
      state->edit_error.clear();
      cache->valid = false;
    } else {
      state->edit_error = error;
    }
    *changed = true;
  }

  if (ImGui::GetContentRegionAvail().x > 96.0f) ImGui::SameLine();
  UndoStack &undo = session.dbc_undo;
  const bool undo_disabled = !undo.can_undo();
  if (undo_disabled) ImGui::BeginDisabled();
  if (ImGui::Button("Undo")) {
    undo.undo();
    cache->valid = false;
    if (message_cache != nullptr) message_cache->valid = false;
    state->edit_error.clear();
    *changed = true;
  }
  if (undo_disabled) ImGui::EndDisabled();
  if (undo.can_undo() && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", undo.undo_text().c_str());

  if (ImGui::GetContentRegionAvail().x > 96.0f) ImGui::SameLine();
  const bool redo_disabled = !undo.can_redo();
  if (redo_disabled) ImGui::BeginDisabled();
  if (ImGui::Button("Redo")) {
    undo.redo();
    cache->valid = false;
    if (message_cache != nullptr) message_cache->valid = false;
    state->edit_error.clear();
    *changed = true;
  }
  if (redo_disabled) ImGui::EndDisabled();
  if (undo.can_redo() && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", undo.redo_text().c_str());

  if (!state->edit_error.empty()) {
    ImGui::SameLine();
    ImGui::TextDisabled("%s", state->edit_error.c_str());
  }
}

struct SignalRowResult {
  bool selected = false;
  bool plot = false;
  bool remove = false;
};

SignalRowResult draw_signal_row(const SignalPaneRow &row, bool selected) {
  SignalRowResult result;
  ImGui::TableNextRow(ImGuiTableRowFlags_None, std::max(ImGui::GetFrameHeight(), ImGui::GetTextLineHeight() + 8.0f));
  // Per-row ID scope so the "plot"/"x" buttons (same labels every row) don't collide.
  ImGui::PushID(row.name.c_str());

  // Per-signal actions (cabana parity): plot the decoded signal, or delete it from the DBC.
  // Only defined DBC signals get actions — a raw bit-candidate has nothing to plot or delete.
  ImGui::TableSetColumnIndex(0);
  if (row.from_dbc) {
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4.0f, 1.0f));
    result.plot = ImGui::SmallButton("plot");
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Plot this signal");
    ImGui::SameLine(0.0f, 3.0f);
    result.remove = ImGui::SmallButton("x");
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Delete this signal");
    ImGui::PopStyleVar();
  }

  ImGui::TableSetColumnIndex(1);
  result.selected = ImGui::Selectable(row.name.c_str(), selected, ImGuiSelectableFlags_SpanAllColumns);

  ImGui::TableSetColumnIndex(2);
  ImGui::TextUnformatted(row.kind.c_str());

  ImGui::TableSetColumnIndex(3);
  push_mono_font();
  ImGui::Text("%d", row.start_bit);
  pop_mono_font();

  ImGui::TableSetColumnIndex(4);
  ImGui::Text("%d", row.size);

  ImGui::TableSetColumnIndex(5);
  ImGui::TextUnformatted(row.endian.c_str());

  ImGui::TableSetColumnIndex(6);
  push_mono_font();
  ImGui::TextUnformatted(row.value.c_str());
  pop_mono_font();

  ImGui::TableSetColumnIndex(7);
  draw_signal_sparkline(row);

  ImGui::TableSetColumnIndex(8);
  if (row.from_dbc || row.byte_change_age < 0.0) ImGui::TextDisabled("--");
  else ImGui::Text("%.2fs", row.byte_change_age);

  ImGui::PopID();
  return result;
}

}  // namespace

void draw_signal_pane(Session &session, PaneInstance &pane) {
  SignalPaneTransientState &transient_state = signal_pane_transient_state(pane);
  SignalPaneState &state = signal_pane_state(pane, transient_state);
  // New/Paste/Open/Close All replace the active DBC file set; any stashed per-signal drafts refer
  // to files that no longer exist, so drop them rather than let them resurface under a coincidental
  // (id, name) match in whatever gets loaded next.
  if (transient_state.dbc_generation != session.dbc.file_set_generation()) {
    transient_state.pending_signal_edits.clear();
    transient_state.dbc_generation = session.dbc.file_set_generation();
  }
  SelectionContext &selection = session.selection(pane.selection_group);
  const std::optional<MessageId> selected = selection.has_selected_msg ? std::optional<MessageId>(selection.selected_msg_id) : std::nullopt;
  MessageId id = initial_message_id_for_store(session.store, pane.state_json, selected);
  if (!selection.has_selected_msg) {
    selection.selected_msg_id = id;
    selection.has_selected_msg = true;
  }

  bool changed = false;
  std::array<char, 128> filter_buf{};
  std::snprintf(filter_buf.data(), filter_buf.size(), "%s", state.filter.c_str());
  const float filter_width = std::clamp(ImGui::GetContentRegionAvail().x * 0.45f, 132.0f, 260.0f);
  ImGui::SetNextItemWidth(filter_width);
  if (ImGui::InputTextWithHint("Filter", "Signal or bit", filter_buf.data(), filter_buf.size())) {
    state.filter = filter_buf.data();
    changed = true;
  }
  if (ImGui::GetContentRegionAvail().x > 132.0f) ImGui::SameLine();
  ImGui::SetNextItemWidth(96.0f);
  if (ImGui::SliderInt("Spark", &state.sparkline_seconds, 1, 120, "%ds", ImGuiSliderFlags_AlwaysClamp)) changed = true;
  if (changed) save_signal_pane_state(pane, transient_state);

  Msg *msg = session.dbc.msg(id);
  // Value/bit-candidates follow the tracker, not the chart's zoom/view range; the sparkline is
  // a separate tracker-anchored window.
  const double tracker_time = session.playback.tracker_time();
  const TimeRange state_range{session.playback.route_range().start_, tracker_time};
  const TimeRange sparkline_range{tracker_time - static_cast<double>(state.sparkline_seconds), tracker_time};
  const std::vector<SignalPaneRow> rows =
      prepare_signal_pane_rows(session.store, id, state_range, sparkline_range, state, msg);
  const bool from_dbc = !rows.empty() && rows.front().from_dbc;
  if (ImGui::GetContentRegionAvail().x > 160.0f) ImGui::SameLine();
  ImGui::TextDisabled("ID %s | %zu %s", id.to_string().c_str(), rows.size(), from_dbc ? "DBC signals" : "bit candidates");

  if (msg != nullptr) {
    draw_message_editor(session, id, &state, &transient_state.message, &transient_state.edit, &changed);
  } else {
    transient_state.message.valid = false;
  }

  if (rows.empty()) {
    ImGui::TextDisabled("No signals or CAN bits in view");
    if (changed) save_signal_pane_state(pane, transient_state);
    return;
  }

  if (from_dbc) {
    const auto selected_it = std::find_if(rows.begin(), rows.end(), [&](const SignalPaneRow &row) {
      return row.name == state.selected_signal;
    });
    if (selected_it == rows.end()) {
      stash_pending_signal_edit(transient_state, id, msg);
      state.selected_signal = rows.front().name;
      if (rows.front().signal == nullptr ||
          !restore_pending_signal_edit(transient_state, id, *rows.front().signal, session.dbc.generation())) {
        transient_state.edit.valid = false;
      }
      changed = true;
    }
  }

  constexpr ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
                                    ImGuiTableFlags_BordersOuter | ImGuiTableFlags_SizingFixedFit |
                                    ImGuiTableFlags_ScrollY;
  // Never ask BeginTable for more height than the pane has left. A short pane can leave avail.y
  // at zero or negative once the message editor above took its share; a positive *minimum* (the
  // old bug) then placed the table's child past the pane's own clip rect, and BeginTable silently
  // returns false for a region that starts off-window (#25). Capping to real avail keeps it
  // on-window (if only a sliver); if there's truly no room, fall through to the signal editor
  // below instead of returning early -- a table render failure must never take it down too.
  ImVec2 table_size = ImGui::GetContentRegionAvail();
  // Bound the table so the editors below stay reachable, but let it grow with the pane —
  // a hard 170px cap forced presets to starve other panes just to show a few signal rows.
  if (from_dbc) {
    const float wanted = std::clamp(table_size.y * 0.45f, 96.0f, 320.0f);
    table_size.y = std::min(wanted, std::max(0.0f, table_size.y));
  }
  if (table_size.y > 0.0f && ImGui::BeginTable("##loggy_signal_table", 9, flags, table_size)) {
    ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 66.0f);
    ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, 138.0f);
    ImGui::TableSetupColumn("Kind", ImGuiTableColumnFlags_WidthFixed, 58.0f);
    ImGui::TableSetupColumn("Start", ImGuiTableColumnFlags_WidthFixed, 48.0f);
    ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthFixed, 42.0f);
    ImGui::TableSetupColumn("Endian", ImGuiTableColumnFlags_WidthFixed, 54.0f);
    ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthFixed, 100.0f);
    ImGui::TableSetupColumn("Spark", ImGuiTableColumnFlags_WidthFixed, 100.0f);
    ImGui::TableSetupColumn("Changed", ImGuiTableColumnFlags_WidthFixed, 58.0f);
    ImGui::TableHeadersRow();

    std::string plot_signal_name, delete_signal_name;
    ImGuiListClipper clipper;
    clipper.Begin(static_cast<int>(rows.size()), std::max(ImGui::GetFrameHeight(), ImGui::GetTextLineHeight() + 8.0f));
    while (clipper.Step()) {
      for (int row_idx = clipper.DisplayStart; row_idx < clipper.DisplayEnd; ++row_idx) {
        const SignalPaneRow &row = rows[static_cast<size_t>(row_idx)];
        const std::string new_selection = row.from_dbc ? row.name : std::string();
        const SignalRowResult row_result = draw_signal_row(row, row.from_dbc && row.name == state.selected_signal);
        if (row_result.plot) plot_signal_name = row.name;
        if (row_result.remove) delete_signal_name = row.name;
        if (row_result.selected &&
            new_selection != state.selected_signal) {
          // Re-clicking the already-selected row is a no-op (ImGui::Selectable reports a click
          // either way); only a genuine switch may touch the edit cache, and even then the
          // previous signal's unapplied edits are stashed rather than dropped.
          stash_pending_signal_edit(transient_state, id, msg);
          state.selected_signal = new_selection;
          state.edit_error.clear();
          if (row.signal == nullptr || !restore_pending_signal_edit(transient_state, id, *row.signal, session.dbc.generation())) {
            transient_state.edit.valid = false;
          }
          changed = true;
        }
      }
    }
    ImGui::EndTable();

    if (!plot_signal_name.empty()) {
      session.plot_decoded_signal(id, plot_signal_name);
    }
    if (!delete_signal_name.empty()) {
      std::string error;
      if (remove_signal_edit(session.dbc_undo, session.dbc, id, delete_signal_name, error)) {
        if (state.selected_signal == delete_signal_name) state.selected_signal.clear();
        state.edit_error.clear();
        transient_state.edit.valid = false;
      } else {
        state.edit_error = error;
      }
      changed = true;
    }
  } else {
    ImGui::Dummy(ImVec2(1.0f, std::max(1.0f, table_size.y)));
  }
  if (from_dbc) {
    draw_signal_editor(session, id, &state, &transient_state.edit, &transient_state.message, &transient_state, &changed);
  }
  if (changed) save_signal_pane_state(pane, transient_state);
}

}  // namespace loggy
