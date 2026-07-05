#include "tools/loggy/backend/session.h"
#include "tools/loggy/panes/signal.h"
#include "tools/loggy/backend/dbc/dbcmanager.h"
#include "tools/loggy/shell/theme.h"
#include "tools/loggy/shell/workspace.h"

#include "imgui.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <optional>
#include <vector>

namespace loggy {
namespace {

void draw_signal_row(const SignalPaneRow &row) {
  ImGui::TableNextRow(ImGuiTableRowFlags_None, std::max(ImGui::GetFrameHeight(), ImGui::GetTextLineHeight() + 8.0f));

  ImGui::TableSetColumnIndex(0);
  ImGui::TextUnformatted(row.name.c_str());

  ImGui::TableSetColumnIndex(1);
  ImGui::TextUnformatted(row.kind.c_str());

  ImGui::TableSetColumnIndex(2);
  push_mono_font();
  ImGui::Text("%d", row.start_bit);
  pop_mono_font();

  ImGui::TableSetColumnIndex(3);
  ImGui::Text("%d", row.size);

  ImGui::TableSetColumnIndex(4);
  ImGui::TextUnformatted(row.endian.c_str());

  ImGui::TableSetColumnIndex(5);
  push_mono_font();
  ImGui::TextUnformatted(row.value.c_str());
  pop_mono_font();

  ImGui::TableSetColumnIndex(6);
  if (row.from_dbc) ImGui::TextUnformatted("--");
  else ImGui::Text("%u", row.flip_count);
}

}  // namespace

void draw_signal_pane(Session &session, PaneInstance &pane) {
  SignalPaneState state = parse_signal_pane_state(pane.state_json);
  SelectionContext &selection = session.selection(pane.selection_group);
  const std::optional<MessageId> selected = selection.has_selected_msg ? std::optional<MessageId>(selection.selected_msg_id) : std::nullopt;
  MessageId id = initial_message_id_for_store(session.store(), pane.state_json, selected);
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
  if (changed) pane.state_json = signal_pane_state_json(state);

  const std::vector<SignalPaneRow> rows = prepare_signal_pane_rows(session.store(), id, session.view_range().range(), state, dbc()->msg(id));
  const bool from_dbc = !rows.empty() && rows.front().from_dbc;
  if (ImGui::GetContentRegionAvail().x > 160.0f) ImGui::SameLine();
  ImGui::TextDisabled("ID %s | %zu %s", id.toString().c_str(), rows.size(), from_dbc ? "DBC signals" : "bit candidates");

  if (rows.empty()) {
    ImGui::TextDisabled("No signals or CAN bits in view");
    return;
  }

  constexpr ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
                                    ImGuiTableFlags_BordersOuter | ImGuiTableFlags_SizingFixedFit |
                                    ImGuiTableFlags_ScrollY;
  if (!ImGui::BeginTable("##loggy_signal_table", 7, flags, ImGui::GetContentRegionAvail())) return;
  ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, 138.0f);
  ImGui::TableSetupColumn("Kind", ImGuiTableColumnFlags_WidthFixed, 58.0f);
  ImGui::TableSetupColumn("Start", ImGuiTableColumnFlags_WidthFixed, 48.0f);
  ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthFixed, 42.0f);
  ImGui::TableSetupColumn("Endian", ImGuiTableColumnFlags_WidthFixed, 54.0f);
  ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthFixed, 100.0f);
  ImGui::TableSetupColumn("Flips", ImGuiTableColumnFlags_WidthFixed, 52.0f);
  ImGui::TableHeadersRow();

  ImGuiListClipper clipper;
  clipper.Begin(static_cast<int>(rows.size()), std::max(ImGui::GetFrameHeight(), ImGui::GetTextLineHeight() + 8.0f));
  while (clipper.Step()) {
    for (int row_idx = clipper.DisplayStart; row_idx < clipper.DisplayEnd; ++row_idx) {
      draw_signal_row(rows[static_cast<size_t>(row_idx)]);
    }
  }
  ImGui::EndTable();
}

}  // namespace loggy
