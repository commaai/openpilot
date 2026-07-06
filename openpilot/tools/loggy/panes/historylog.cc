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

HistoryLogParams history_log_params_from_state(const HistoryLogState &state) {
  return {
    .filter = state.filter,
    .compare_signal = state.compare_signal,
    .compare_op = state.compare_op,
    .compare_value = state.compare_value,
    .compare_enabled = state.compare_enabled,
    .max_rows = state.max_rows,
    .page_size = state.page_size,
    .page_index = state.page_index,
  };
}

HistoryLogPage prepare_history_log_page(const Store &store, const MessageId &id, TimeRange range,
                                       const HistoryLogState &state, const Msg *msg = nullptr) {
  return prepare_history_log_page(store, id, range, history_log_params_from_state(state), msg);
}

struct HistoryLogPaneTransientState {
  std::string state_json;
  HistoryLogState state;
  MessageId active_id = kDefaultLoggyMessageId;
  // Building a page copies every event in range (24k+ on an rlog message); cache it and rebuild
  // only when the inputs actually change, not per frame.
  HistoryLogPage page;
  std::string page_key;
  uint64_t page_store_gen = UINT64_MAX;
  uint64_t page_dbc_gen = UINT64_MAX;
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

  // Rows show events up to the tracker, newest-first from the playhead (cabana semantics); a
  // chart zoom must not affect this pane. Copy/Save CSV export the FULL route instead -- that
  // matches Qt cabana's exports, and the difference from the on-screen page is intentional.
  const double tracker_time = session.playback.tracker_time();
  const TimeRange page_range{session.playback.route_range().start_, tracker_time};
  const TimeRange export_range = session.playback.route_range();

  // Rebuilding the page copies every matching event, so it must not run every frame just because
  // the tracker ticked forward. can_event_summary is O(log n); its count/last_time only change
  // when an event actually crosses the tracker or a seek moves it, so they stand in for the raw
  // tracker time in the cache key.
  const CanSummaryView playhead = session.store.can_event_summary(id, page_range, /*with_data=*/false);
  const auto page_cache_key = [&]() {
    char buf[192];
    std::snprintf(buf, sizeof(buf), "%s|%zu|%.6f|%zu|%zu|%d|%.9g|%s|%s|%s", id.to_string().c_str(),
                  playhead.count, playhead.last_time, state.page_size, state.page_index, state.compare_enabled ? 1 : 0,
                  state.compare_value, state.compare_signal.c_str(), state.compare_op.c_str(), state.filter.c_str());
    return std::string(buf);
  };
  const auto refresh_page = [&]() {
    transient.page = prepare_history_log_page(session.store, id, page_range, state, msg);
    transient.page_key = page_cache_key();
    transient.page_store_gen = session.store.generation();
    transient.page_dbc_gen = session.dbc.generation();
  };
  if (transient.page_key != page_cache_key() || transient.page_store_gen != session.store.generation() ||
      transient.page_dbc_gen != session.dbc.generation()) {
    refresh_page();
  }
  HistoryLogPage &page = transient.page;
  if (page.page_index != state.page_index) {
    state.page_index = page.page_index;
    changed = true;
  }

  // page.total_rows is the match count, capped at max_rows; report it against the id's TRUE
  // event count up to the tracker (`playhead`, unfiltered) rather than let the cap silently
  // pass for the whole truth (REVIEW.md defect #31 — "N events" used to cap at max_rows).
  const size_t true_total = playhead.count;
  char events_label[96];
  if (page.truncated) {
    std::snprintf(events_label, sizeof(events_label), "showing first %zu of %zu events", page.total_rows, true_total);
  } else if (page.total_rows != true_total) {
    std::snprintf(events_label, sizeof(events_label), "%zu matching of %zu events", page.total_rows, true_total);
  } else {
    std::snprintf(events_label, sizeof(events_label), "%zu events", page.total_rows);
  }
  if (ImGui::GetContentRegionAvail().x > 160.0f) ImGui::SameLine();
  ImGui::TextDisabled("ID %s | %s", id.to_string().c_str(), events_label);
  if (ImGui::GetContentRegionAvail().x > 92.0f) ImGui::SameLine();
  if (ImGui::Button("Copy CSV")) {
    const std::string csv = can_message_csv(session.store, id, export_range, msg);
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
    changed = save_history_csv(&state, can_message_csv(session.store, id, export_range, msg), "");
  }
  if (ImGui::GetContentRegionAvail().x > 126.0f) ImGui::SameLine();
  if (ImGui::Button("Save Stream")) {
    changed = save_history_csv(&state, can_stream_csv(session.store, export_range), "_stream");
  }
  const Signal *export_sig = history_export_signal(state, msg);
  if (ImGui::GetContentRegionAvail().x > 118.0f) ImGui::SameLine();
  ImGui::BeginDisabled(export_sig == nullptr);
  if (ImGui::Button("Save Signal") && export_sig != nullptr) {
    changed = save_history_csv(&state, can_signal_csv(session.store, id, export_range, *export_sig), "_" + export_sig->name);
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
    refresh_page();
  }
  if (ImGui::GetContentRegionAvail().x > 188.0f) ImGui::SameLine();
  const bool can_prev = page.page_index > 0;
  const bool can_next = page.page_index + 1 < page.page_count;
  ImGui::BeginDisabled(!can_prev);
  if (ImGui::Button("<")) {
    --state.page_index;
    changed = true;
    refresh_page();
  }
  ImGui::EndDisabled();
  ImGui::SameLine();
  ImGui::TextDisabled("%zu/%zu", page.page_index + 1, page.page_count);
  ImGui::SameLine();
  ImGui::BeginDisabled(!can_next);
  if (ImGui::Button(">")) {
    ++state.page_index;
    changed = true;
    refresh_page();
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
