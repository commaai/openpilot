#include "tools/loggy/backend/session.h"
#include "tools/loggy/backend/export.h"
#include "tools/loggy/panes/historylog.h"
#include "tools/loggy/backend/dbc/dbcmanager.h"
#include "tools/loggy/shell/theme.h"
#include "tools/loggy/shell/workspace.h"

#include "imgui.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

namespace loggy {
namespace {

constexpr std::array<const char *, 6> kHistoryCompareOps = {">", "=", "!=", "<", ">=", "<="};

const Signal *history_export_signal(const HistoryLogState &state, const Msg *msg) {
  if (msg == nullptr || msg->getSignals().empty()) return nullptr;
  if (!state.compare_signal.empty()) {
    if (const Signal *sig = msg->sig(state.compare_signal)) return sig;
  }
  return msg->getSignals().front();
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
  if (!write_csv_file(path, csv, &error)) {
    state->export_status = "Export failed: " + error;
  } else {
    state->export_path = path.string();
    state->export_status = "Saved " + path.string();
  }
  return true;
}

bool draw_compare_controls(HistoryLogState *state, const Msg *msg) {
  if (msg == nullptr || msg->getSignals().empty()) return false;

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
    for (const Signal *sig : msg->getSignals()) {
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
  HistoryLogState state = parse_history_log_state(pane.state_json);
  SelectionContext &selection = session.selection(pane.selection_group);
  const std::optional<MessageId> selected = selection.has_selected_msg ? std::optional<MessageId>(selection.selected_msg_id) : std::nullopt;
  MessageId id = initial_message_id_for_store(session.store(), pane.state_json, selected);
  if (!selection.has_selected_msg) {
    selection.selected_msg_id = id;
    selection.has_selected_msg = true;
  }

  bool changed = false;
  bool filter_changed = false;
  const Msg *msg = dbc()->msg(id);
  if (state.compare_enabled && (msg == nullptr || msg->sig(state.compare_signal) == nullptr)) {
    state.compare_enabled = false;
    state.compare_signal.clear();
    changed = true;
    filter_changed = true;
  }
  std::array<char, 128> filter_buf{};
  std::snprintf(filter_buf.data(), filter_buf.size(), "%s", state.filter.c_str());
  const float filter_width = std::clamp(ImGui::GetContentRegionAvail().x * 0.42f, 132.0f, 260.0f);
  ImGui::SetNextItemWidth(filter_width);
  if (ImGui::InputTextWithHint("Filter", "Hex or decoded", filter_buf.data(), filter_buf.size())) {
    state.filter = filter_buf.data();
    changed = true;
    filter_changed = true;
  }
  if (draw_compare_controls(&state, msg)) {
    changed = true;
    filter_changed = true;
  }
  if (filter_changed) state.page_index = 0;

  const TimeRange range = session.view_range().range();
  HistoryLogPage page = prepare_history_log_page(session.store(), id, range, state, msg);
  if (page.page_index != state.page_index) {
    state.page_index = page.page_index;
    changed = true;
  }

  if (ImGui::GetContentRegionAvail().x > 160.0f) ImGui::SameLine();
  ImGui::TextDisabled("ID %s | %zu events%s", id.toString().c_str(), page.total_rows, page.truncated ? "+" : "");
  if (ImGui::GetContentRegionAvail().x > 92.0f) ImGui::SameLine();
  if (ImGui::Button("Copy CSV")) {
    const std::string csv = can_message_csv(session.store(), id, range, msg);
    ImGui::SetClipboardText(csv.c_str());
  }

  std::array<char, 256> export_buf{};
  std::snprintf(export_buf.data(), export_buf.size(), "%s", state.export_path.c_str());
  ImGui::SetNextItemWidth(std::clamp(ImGui::GetContentRegionAvail().x * 0.42f, 180.0f, 360.0f));
  if (ImGui::InputTextWithHint("Export", "/tmp/loggy_history.csv", export_buf.data(), export_buf.size())) {
    state.export_path = export_buf.data();
    state.export_status.clear();
    changed = true;
  }
  if (ImGui::GetContentRegionAvail().x > 108.0f) ImGui::SameLine();
  if (ImGui::Button("Save Msg")) {
    changed = save_history_csv(&state, can_message_csv(session.store(), id, range, msg), "");
  }
  if (ImGui::GetContentRegionAvail().x > 126.0f) ImGui::SameLine();
  if (ImGui::Button("Save Stream")) {
    changed = save_history_csv(&state, can_stream_csv(session.store(), range), "_stream");
  }
  const Signal *export_sig = history_export_signal(state, msg);
  if (ImGui::GetContentRegionAvail().x > 118.0f) ImGui::SameLine();
  ImGui::BeginDisabled(export_sig == nullptr);
  if (ImGui::Button("Save Signal") && export_sig != nullptr) {
    changed = save_history_csv(&state, can_signal_csv(session.store(), id, range, *export_sig), "_" + export_sig->name);
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
    page = prepare_history_log_page(session.store(), id, range, state, msg);
  }
  if (ImGui::GetContentRegionAvail().x > 188.0f) ImGui::SameLine();
  const bool can_prev = page.page_index > 0;
  const bool can_next = page.page_index + 1 < page.page_count;
  ImGui::BeginDisabled(!can_prev);
  if (ImGui::Button("<")) {
    --state.page_index;
    changed = true;
    page = prepare_history_log_page(session.store(), id, range, state, msg);
  }
  ImGui::EndDisabled();
  ImGui::SameLine();
  ImGui::TextDisabled("%zu/%zu", page.page_index + 1, page.page_count);
  ImGui::SameLine();
  ImGui::BeginDisabled(!can_next);
  if (ImGui::Button(">")) {
    ++state.page_index;
    changed = true;
    page = prepare_history_log_page(session.store(), id, range, state, msg);
  }
  ImGui::EndDisabled();

  if (changed) pane.state_json = history_log_state_json(id, state);

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
