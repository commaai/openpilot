#include "tools/loggy/backend/csv.h"
#include "tools/loggy/backend/session.h"
#include "tools/loggy/panes/historylog.h"
#include "tools/loggy/backend/dbc/dbcmanager.h"
#include "tools/loggy/shell/native_dialog.h"
#include "tools/loggy/shell/theme.h"
#include "tools/loggy/shell/workspace.h"

#include "json11/json11.hpp"

#include "imgui.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <cctype>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace loggy {
namespace {

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

bool history_valid_compare_op(std::string_view op) {
  return op == ">" || op == "=" || op == "!=" || op == "<" || op == ">=" || op == "<=";
}

HistoryLogState parse_history_log_state(std::string_view state_json) {
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

std::string history_log_state_json(const MessageId &id, const HistoryLogState &state) {
  return json11::Json(json11::Json::object{
    {"id", id.to_string()},
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

std::string history_lower_text(std::string_view text) {
  std::string out(text);
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}

bool history_text_matches_filter(std::string_view text, std::string_view filter) {
  return filter.empty() || history_lower_text(text).find(history_lower_text(filter)) != std::string::npos;
}

bool history_compare_values(double lhs, std::string_view op, double rhs) {
  if (op == ">") return lhs > rhs;
  if (op == "=") return lhs == rhs;
  if (op == "!=") return lhs != rhs;
  if (op == "<") return lhs < rhs;
  if (op == ">=") return lhs >= rhs;
  if (op == "<=") return lhs <= rhs;
  return true;
}

std::string history_hex_bytes(const std::vector<uint8_t> &bytes) {
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

std::string history_decoded_values(const Msg *msg, const std::vector<uint8_t> &data, size_t max_values = 4) {
  if (msg == nullptr || data.empty()) return {};
  std::string out;
  size_t count = 0;
  for (const Signal *sig : msg->signals()) {
    if (sig == nullptr) continue;
    double value = 0.0;
    if (!sig->get_value(data.data(), data.size(), &value)) continue;
    if (!out.empty()) out += ", ";
    out += sig->name + "=" + sig->format_value(value);
    if (++count >= max_values) break;
  }
  return out;
}

bool history_matches_compare(const Msg *msg, const HistoryLogState &state, const std::vector<uint8_t> &data) {
  if (!state.compare_enabled || state.compare_signal.empty()) return true;
  if (msg == nullptr || data.empty()) return false;
  const Signal *sig = msg->sig(state.compare_signal);
  if (sig == nullptr) return false;
  double value = 0.0;
  if (!sig->get_value(data.data(), data.size(), &value)) return false;
  return history_compare_values(value, state.compare_op, state.compare_value);
}

HistoryLogPage prepare_history_log_page(const Store &store,
                                       const MessageId &id,
                                       TimeRange range,
                                       const HistoryLogState &state,
                                       const Msg *msg = nullptr) {
  const CanEventView view = store.can_events(id, range);
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

  const size_t start_ = std::min(matches.size(), page.page_index * page.page_size);
  const size_t end = std::min(matches.size(), start_ + page.page_size);
  page.rows.assign(matches.begin() + static_cast<std::ptrdiff_t>(start_), matches.begin() + static_cast<std::ptrdiff_t>(end));
  return page;
}

struct HistoryLogPaneTransientState {
  std::string state_json;
  HistoryLogState state;
  MessageId active_id = kDefaultLoggyMessageId;
};

constexpr std::array<const char *, 6> kHistoryCompareOps = {">", "=", "!=", "<", ">=", "<="};

const Signal *history_export_signal(const HistoryLogState &state, const Msg *msg) {
  if (msg == nullptr || msg->signals().empty()) return nullptr;
  if (!state.compare_signal.empty()) {
    if (const Signal *sig = msg->sig(state.compare_signal)) return sig;
  }
  return msg->signals().front();
}

bool choose_history_export_path(HistoryLogState *state) {
  if (state == nullptr) return false;
  std::string error;
  const std::optional<std::string> path = native_dialog_choose_path(
    NativeDialogType::SaveFile,
    {.title = "Save CAN CSV", .path = state->export_path, .confirm_overwrite = true},
    error);
  if (path.has_value()) {
    state->export_path = *path;
    state->export_status.clear();
    return true;
  }
  state->export_status = error.empty() ? "Export path unchanged" : "Dialog failed: " + error;
  return true;
}

bool save_history_csv(HistoryLogState *state, std::string csv, std::string suffix) {
  std::filesystem::path path(state->export_path);
  if (path.empty()) {
    state->export_status = "Export failed: empty path";
    return true;
  }
  if (!suffix.empty()) {
    path.replace_filename(path.stem().string() + suffix + ".csv");
  } else if (path.extension().empty()) {
    path += ".csv";
  }
  std::string error;
  if (!write_csv_file(path, csv, error)) {
    state->export_status = "Export failed: " + error;
  } else {
    state->export_path = path.string();
    state->export_status = "Saved " + path.string();
  }
  return true;
}

bool draw_compare_controls(HistoryLogState *state, const Msg *msg) {
  if (msg == nullptr || msg->signals().empty()) return false;

  bool changed = false;
  ImGui::SameLine();
  ImGui::SetNextItemWidth(128.0f);
  const char *preview = state->compare_enabled && !state->compare_signal.empty() ? state->compare_signal.c_str() : "Any";
  if (ImGui::BeginCombo("Signal", preview)) {
    if (ImGui::Selectable("Any", !state->compare_enabled)) {
      state->compare_enabled = false;
      state->compare_signal.clear();
      changed = true;
    }
    if (!state->compare_enabled) ImGui::SetItemDefaultFocus();
    for (const Signal *sig : msg->signals()) {
      if (sig == nullptr) continue;
      const bool selected = state->compare_enabled && state->compare_signal == sig->name;
      if (ImGui::Selectable(sig->name.c_str(), selected)) {
        state->compare_signal = sig->name;
        state->compare_enabled = true;
        changed = true;
      }
      if (selected) ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }

  if (!state->compare_enabled) return changed;

  ImGui::SameLine();
  ImGui::SetNextItemWidth(58.0f);
  const char *op_preview = history_valid_compare_op(state->compare_op) ? state->compare_op.c_str() : ">";
  if (ImGui::BeginCombo("Op", op_preview)) {
    for (const char *op : kHistoryCompareOps) {
      const bool selected = state->compare_op == op;
      if (ImGui::Selectable(op, selected)) {
        state->compare_op = op;
        changed = true;
      }
      if (selected) ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }

  ImGui::SameLine();
  ImGui::SetNextItemWidth(104.0f);
  if (ImGui::InputDouble("Value", &state->compare_value, 0.0, 0.0, "%.6g")) {
    changed = true;
  }
  return changed;
}

HistoryLogPaneTransientState &history_log_pane_transient_state(PaneInstance &pane) {
  if (HistoryLogPaneTransientState *state = std::any_cast<HistoryLogPaneTransientState>(&pane.transient_state)) {
    return *state;
  }
  pane.transient_state = HistoryLogPaneTransientState{};
  return std::any_cast<HistoryLogPaneTransientState &>(pane.transient_state);
}

HistoryLogState &history_log_pane_state(const Store &store, PaneInstance &pane) {
  HistoryLogPaneTransientState &transient = history_log_pane_transient_state(pane);
  if (transient.state_json != pane.state_json) {
    transient.state = parse_history_log_state(pane.state_json);
    transient.active_id = initial_message_id_for_store(store, pane.state_json, std::nullopt);
    transient.state_json = pane.state_json;
  }
  return transient.state;
}

void draw_history_row(const HistoryLogRow &row) {
  ImGui::TableNextRow(ImGuiTableRowFlags_None, std::max(ImGui::GetFrameHeight(), ImGui::GetTextLineHeight() + 8.0f));

  ImGui::TableSetColumnIndex(0);
  push_mono_font();
  ImGui::Text("%.3f", row.mono_time);
  pop_mono_font();

  ImGui::TableSetColumnIndex(1);
  ImGui::Text("%u", static_cast<unsigned>(row.bus_time));

  ImGui::TableSetColumnIndex(2);
  ImGui::Text("%zu", row.byte_count);

  ImGui::TableSetColumnIndex(3);
  push_mono_font();
  ImGui::TextUnformatted(row.data_hex.c_str());
  pop_mono_font();

  ImGui::TableSetColumnIndex(4);
  if (row.decoded.empty()) ImGui::TextDisabled("--");
  else ImGui::TextUnformatted(row.decoded.c_str());
}

}  // namespace

void draw_history_log_pane(Session &session, PaneInstance &pane) {
  HistoryLogPaneTransientState &transient = history_log_pane_transient_state(pane);
  HistoryLogState &state = history_log_pane_state(session.store, pane);
  SelectionContext &selection = session.selection(pane.selection_group);
  const std::optional<MessageId> selected = selection.has_selected_msg ? std::optional<MessageId>(selection.selected_msg_id) : std::nullopt;
  MessageId id = selected.has_value() ? *selected : transient.active_id;
  transient.active_id = id;
  if (!selection.has_selected_msg) {
    selection.selected_msg_id = id;
    selection.has_selected_msg = true;
  }

  bool changed = false;
  bool filter_changed = false;
  const Msg *msg = session.dbc.msg(id);
  if (state.compare_enabled && (msg == nullptr || msg->sig(state.compare_signal) == nullptr)) {
    state.compare_enabled = false;
    state.compare_signal.clear();
    changed = true;
    filter_changed = true;
  }
  const float filter_width = std::clamp(ImGui::GetContentRegionAvail().x * 0.42f, 132.0f, 260.0f);
  ImGui::SetNextItemWidth(filter_width);
  if (input_text_with_hint("Filter", "Hex or decoded", &state.filter)) {
    changed = true;
    filter_changed = true;
  }
  if (draw_compare_controls(&state, msg)) {
    changed = true;
    filter_changed = true;
  }
  if (filter_changed) state.page_index = 0;

  const TimeRange range = session.view_range.range();
  HistoryLogPage page = prepare_history_log_page(session.store, id, range, state, msg);
  if (page.page_index != state.page_index) {
    state.page_index = page.page_index;
    changed = true;
  }

  if (ImGui::GetContentRegionAvail().x > 160.0f) ImGui::SameLine();
  ImGui::TextDisabled("ID %s | %zu events%s", id.to_string().c_str(), page.total_rows, page.truncated ? "+" : "");
  if (ImGui::GetContentRegionAvail().x > 92.0f) ImGui::SameLine();
  if (ImGui::Button("Copy CSV")) {
    const std::string csv = can_message_csv(session.store, id, range, msg);
    ImGui::SetClipboardText(csv.c_str());
  }

  ImGui::SetNextItemWidth(std::clamp(ImGui::GetContentRegionAvail().x * 0.42f, 180.0f, 360.0f));
  if (input_text_with_hint("Export", "/tmp/loggy_history.csv", &state.export_path)) {
    state.export_status.clear();
    changed = true;
  }
  if (ImGui::GetContentRegionAvail().x > 92.0f) ImGui::SameLine();
  if (ImGui::Button("Browse##history_export")) {
    changed = choose_history_export_path(&state) || changed;
  }
  if (ImGui::GetContentRegionAvail().x > 108.0f) ImGui::SameLine();
  if (ImGui::Button("Save Msg")) {
    changed = save_history_csv(&state, can_message_csv(session.store, id, range, msg), "");
  }
  if (ImGui::GetContentRegionAvail().x > 126.0f) ImGui::SameLine();
  if (ImGui::Button("Save Stream")) {
    changed = save_history_csv(&state, can_stream_csv(session.store, range), "_stream");
  }
  const Signal *export_sig = history_export_signal(state, msg);
  if (ImGui::GetContentRegionAvail().x > 118.0f) ImGui::SameLine();
  ImGui::BeginDisabled(export_sig == nullptr);
  if (ImGui::Button("Save Signal") && export_sig != nullptr) {
    changed = save_history_csv(&state, can_signal_csv(session.store, id, range, *export_sig), "_" + export_sig->name);
  }
  ImGui::EndDisabled();
  if (!state.export_status.empty()) {
    ImGui::SameLine();
    ImGui::TextDisabled("%s", state.export_status.c_str());
  }

  int page_size = static_cast<int>(page.page_size);
  ImGui::SetNextItemWidth(74.0f);
  if (ImGui::InputInt("Rows", &page_size, 0, 0)) {
    state.page_size = static_cast<size_t>(std::clamp(page_size, 1, 5000));
    state.page_index = 0;
    changed = true;
    page = prepare_history_log_page(session.store, id, range, state, msg);
  }
  if (ImGui::GetContentRegionAvail().x > 188.0f) ImGui::SameLine();
  const bool can_prev = page.page_index > 0;
  const bool can_next = page.page_index + 1 < page.page_count;
  ImGui::BeginDisabled(!can_prev);
  if (ImGui::Button("<")) {
    --state.page_index;
    changed = true;
    page = prepare_history_log_page(session.store, id, range, state, msg);
  }
  ImGui::EndDisabled();
  ImGui::SameLine();
  ImGui::TextDisabled("%zu/%zu", page.page_index + 1, page.page_count);
  ImGui::SameLine();
  ImGui::BeginDisabled(!can_next);
  if (ImGui::Button(">")) {
    ++state.page_index;
    changed = true;
    page = prepare_history_log_page(session.store, id, range, state, msg);
  }
  ImGui::EndDisabled();

  if (changed) pane.state_json = history_log_state_json(id, state);
  if (changed) transient.state_json = pane.state_json;
  transient.active_id = id;

  if (page.rows.empty()) {
    ImGui::TextDisabled("No CAN history in view or filter");
    return;
  }

  constexpr ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
                                    ImGuiTableFlags_BordersOuter | ImGuiTableFlags_SizingFixedFit |
                                    ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY;
  if (!ImGui::BeginTable("##loggy_history_log", 5, flags, ImGui::GetContentRegionAvail())) return;
  ImGui::TableSetupColumn("Time", ImGuiTableColumnFlags_WidthFixed, 72.0f);
  ImGui::TableSetupColumn("Bus T", ImGuiTableColumnFlags_WidthFixed, 54.0f);
  ImGui::TableSetupColumn("Len", ImGuiTableColumnFlags_WidthFixed, 40.0f);
  ImGui::TableSetupColumn("Hex", ImGuiTableColumnFlags_WidthFixed, 210.0f);
  ImGui::TableSetupColumn("Decoded", ImGuiTableColumnFlags_WidthFixed, 260.0f);
  ImGui::TableHeadersRow();

  ImGuiListClipper clipper;
  clipper.Begin(static_cast<int>(page.rows.size()), std::max(ImGui::GetFrameHeight(), ImGui::GetTextLineHeight() + 8.0f));
  while (clipper.Step()) {
    for (int row_idx = clipper.DisplayStart; row_idx < clipper.DisplayEnd; ++row_idx) {
      draw_history_row(page.rows[static_cast<size_t>(row_idx)]);
    }
  }
  ImGui::EndTable();
}

}  // namespace loggy
