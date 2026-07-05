#include "tools/loggy/panes/messages.h"

#include "tools/loggy/backend/session.h"
#include "tools/loggy/shell/theme.h"
#include "tools/loggy/shell/workspace.h"

#include "imgui.h"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cstdio>
#include <string>
#include <vector>

namespace loggy {
namespace {

void draw_bytes(const std::vector<uint8_t> &bytes) {
  push_mono_font();
  const ImVec2 cell(ImGui::CalcTextSize("00").x + 8.0f, ImGui::GetTextLineHeight() + 4.0f);
  const ImVec2 origin = ImGui::GetCursorScreenPos();
  ImDrawList *draw_list = ImGui::GetWindowDrawList();
  const ImU32 text = ImGui::GetColorU32(ImGuiCol_Text);
  const ImU32 fill = ImGui::GetColorU32(color_rgb(76, 80, 82));
  for (size_t i = 0; i < bytes.size(); ++i) {
    const ImVec2 p0(origin.x + static_cast<float>(i) * cell.x, origin.y);
    const ImVec2 p1(p0.x + cell.x, p0.y + cell.y);
    draw_list->AddRectFilled(p0, p1, fill);
    char buf[3];
    std::snprintf(buf, sizeof(buf), "%02X", bytes[i]);
    const ImVec2 text_size = ImGui::CalcTextSize(buf);
    draw_list->AddText(ImVec2(p0.x + (cell.x - text_size.x) * 0.5f, p0.y + (cell.y - text_size.y) * 0.5f), text, buf);
  }
  ImGui::Dummy(ImVec2(std::max(1.0f, static_cast<float>(bytes.size()) * cell.x), cell.y));
  pop_mono_font();
}

std::vector<int> collect_message_buses(const std::vector<MessageId> &ids) {
  std::vector<int> buses;
  buses.reserve(ids.size());
  for (const MessageId &id : ids) {
    buses.push_back(static_cast<int>(id.source));
  }
  std::sort(buses.begin(), buses.end());
  buses.erase(std::unique(buses.begin(), buses.end()), buses.end());
  return buses;
}

bool draw_bus_filter_combo(MessageTableState *state, const std::vector<int> &buses) {
  const std::string preview = state->bus_filter < 0 ? "All" : std::to_string(state->bus_filter);
  ImGui::SetNextItemWidth(86.0f);
  bool changed = false;
  if (!ImGui::BeginCombo("Bus", preview.c_str())) return false;

  if (ImGui::Selectable("All", state->bus_filter < 0)) {
    state->bus_filter = -1;
    changed = true;
  }
  if (state->bus_filter < 0) ImGui::SetItemDefaultFocus();

  for (const int bus : buses) {
    char label[16];
    std::snprintf(label, sizeof(label), "%d", bus);
    const bool selected = state->bus_filter == bus;
    if (ImGui::Selectable(label, selected)) {
      state->bus_filter = bus;
      changed = true;
    }
    if (selected) ImGui::SetItemDefaultFocus();
  }

  ImGui::EndCombo();
  return changed;
}

void draw_message_row(const MessageTableRow &row, const MessageTableState &state,
                      SelectionContext *selection, PaneInstance *pane) {
  const MessageSummary &summary = row.summary;
  const bool selected_row = selection->has_selected_msg && selection->selected_msg_id == row.id;
  char name[48];
  std::snprintf(name, sizeof(name), "CAN %s", row.id.toString().c_str());

  ImGui::TableNextRow(ImGuiTableRowFlags_None, std::max(ImGui::GetFrameHeight(), ImGui::GetTextLineHeight() + 8.0f));
  ImGui::PushID(static_cast<int>(row.id.source));
  ImGui::PushID(static_cast<int>(row.id.address));

  ImGui::TableSetColumnIndex(0);
  if (ImGui::Selectable(name, selected_row, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap)) {
    selection->selected_msg_id = row.id;
    selection->has_selected_msg = true;
    pane->state_json = message_table_state_json(row.id, state);
  }

  ImGui::TableSetColumnIndex(1);
  ImGui::Text("%u", static_cast<unsigned>(row.id.source));

  ImGui::TableSetColumnIndex(2);
  push_mono_font();
  ImGui::Text("0x%X", row.id.address);
  pop_mono_font();

  ImGui::TableSetColumnIndex(3);
  if (summary.count > 1) ImGui::Text("%.2f", summary.frequency_hz);
  else ImGui::TextUnformatted("--");

  ImGui::TableSetColumnIndex(4);
  ImGui::Text("%zu", summary.count);

  ImGui::TableSetColumnIndex(5);
  if (!summary.latest_data.empty()) {
    draw_bytes(summary.latest_data);
  } else {
    ImGui::TextDisabled("No events in view");
  }

  ImGui::PopID();
  ImGui::PopID();
}

}  // namespace

void draw_messages_pane(Session &session, PaneInstance &pane) {
  SelectionContext &selection = session.selection(pane.selection_group);
  MessageTableState state = parse_message_table_state(pane.state_json);
  const std::optional<MessageId> selected = selection.has_selected_msg ? std::optional<MessageId>(selection.selected_msg_id) : std::nullopt;
  MessageId active_id = initial_message_id_for_store(session.store(), pane.state_json, selected);
  const std::vector<MessageId> all_ids = session.store().canMessageIds();
  const std::vector<int> buses = collect_message_buses(all_ids);

  bool controls_changed = false;
  std::array<char, 128> filter_buf{};
  std::snprintf(filter_buf.data(), filter_buf.size(), "%s", state.filter.c_str());
  const float filter_width = std::clamp(ImGui::GetContentRegionAvail().x * 0.45f, 132.0f, 260.0f);
  ImGui::SetNextItemWidth(filter_width);
  if (ImGui::InputTextWithHint("Filter", "ID", filter_buf.data(), filter_buf.size())) {
    state.filter = filter_buf.data();
    controls_changed = true;
  }

  if (ImGui::GetContentRegionAvail().x > 116.0f) ImGui::SameLine();
  controls_changed = draw_bus_filter_combo(&state, buses) || controls_changed;

  std::vector<MessageTableRow> rows = prepare_message_table_rows(session.store(), session.view_range().range(), state);
  const auto active_row = std::find_if(rows.begin(), rows.end(), [&](const MessageTableRow &row) {
    return row.id == active_id;
  });
  if (!rows.empty() && (!selection.has_selected_msg || active_row == rows.end())) {
    active_id = rows.front().id;
    selection.selected_msg_id = active_id;
    selection.has_selected_msg = true;
    pane.state_json = message_table_state_json(active_id, state);
  } else if (selection.has_selected_msg) {
    active_id = selection.selected_msg_id;
  }
  if (controls_changed) pane.state_json = message_table_state_json(active_id, state);

  if (ImGui::GetContentRegionAvail().x > 142.0f) ImGui::SameLine();
  ImGui::TextDisabled("%zu/%zu CAN ids", rows.size(), all_ids.size());

  constexpr ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
                                    ImGuiTableFlags_BordersOuter | ImGuiTableFlags_SizingFixedFit |
                                    ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY;
  if (rows.empty()) {
    ImGui::TextDisabled("No CAN messages in view or filter");
    return;
  }

  size_t max_bytes = 0;
  for (const MessageTableRow &row : rows) {
    max_bytes = std::max(max_bytes, row.summary.latest_data.size());
  }
  if (!ImGui::BeginTable("##loggy_messages", 6, flags, ImGui::GetContentRegionAvail())) return;
  ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, 126.0f);
  ImGui::TableSetupColumn("Bus", ImGuiTableColumnFlags_WidthFixed, 42.0f);
  ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, 72.0f);
  ImGui::TableSetupColumn("Freq", ImGuiTableColumnFlags_WidthFixed, 64.0f);
  ImGui::TableSetupColumn("Count", ImGuiTableColumnFlags_WidthFixed, 70.0f);
  ImGui::TableSetupColumn("Bytes", ImGuiTableColumnFlags_WidthFixed,
                          std::max(160.0f, std::min(420.0f, static_cast<float>(max_bytes) * 30.0f)));
  ImGui::TableHeadersRow();

  ImGuiListClipper clipper;
  clipper.Begin(static_cast<int>(rows.size()), std::max(ImGui::GetFrameHeight(), ImGui::GetTextLineHeight() + 8.0f));
  while (clipper.Step()) {
    for (int row_idx = clipper.DisplayStart; row_idx < clipper.DisplayEnd; ++row_idx) {
      draw_message_row(rows[static_cast<size_t>(row_idx)], state, &selection, &pane);
    }
  }
  ImGui::EndTable();
}

}  // namespace loggy
