#include "tools/loggy/panes/binary.h"

#include "tools/loggy/backend/session.h"
#include "tools/loggy/backend/dbc/dbcmanager.h"
#include "tools/loggy/backend/undo.h"
#include "tools/loggy/shell/theme.h"
#include "tools/loggy/shell/workspace.h"

#include "imgui.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

namespace loggy {
namespace {

struct BinaryDragState {
  bool active = false;
  bool dragged = false;
  MessageId id;
  int anchor_bit = -1;
  int current_bit = -1;
  std::string status;
};

BinaryDragState &binary_drag_state() {
  static BinaryDragState state;
  return state;
}

ImU32 heat_color(uint32_t flips, uint32_t max_flips) {
  if (flips == 0 || max_flips == 0) return ImGui::GetColorU32(color_rgb(68, 71, 73));
  const float alpha = 0.18f + 0.58f * (std::log2(1.0f + static_cast<float>(flips)) /
                                       std::log2(1.0f + static_cast<float>(max_flips)));
  return ImGui::GetColorU32(color_rgb(47, 101, 202, alpha));
}

}  // namespace

void draw_binary_pane(Session &session, PaneInstance &pane) {
  SelectionContext &selection = session.selection(pane.selection_group);
  const std::optional<MessageId> selected = selection.has_selected_msg ? std::optional<MessageId>(selection.selected_msg_id) : std::nullopt;
  const MessageId id = parse_message_id_state(pane.state_json, selected);
  const std::optional<BinaryGrid> maybe_grid = build_binary_grid(session.store(), id, session.view_range().range());

  ImGui::TextDisabled("ID %s", id.toString().c_str());
  if (!maybe_grid.has_value()) {
    ImGui::TextDisabled("No CAN events in view");
    return;
  }

  const BinaryGrid &grid = *maybe_grid;
  ImGui::SameLine();
  ImGui::TextDisabled("| %zu events | latest %.3fs", grid.event_count, grid.last_time);
  BinaryDragState &drag = binary_drag_state();
  if (!drag.status.empty()) {
    ImGui::SameLine();
    ImGui::TextDisabled("%s", drag.status.c_str());
  }

  constexpr int kBitColumns = 8;
  constexpr int kColumns = kBitColumns + 1;
  const float row_header_w = 30.0f;
  const float row_h = std::max(22.0f, ImGui::GetTextLineHeight() + 8.0f);
  const float avail_w = ImGui::GetContentRegionAvail().x;
  const float col_w = std::max(22.0f, (avail_w - row_header_w) / static_cast<float>(kColumns));
  const float total_w = row_header_w + col_w * static_cast<float>(kColumns);
  const float total_h = row_h * static_cast<float>(grid.rows.size() + 1);
  const ImVec2 origin = ImGui::GetCursorScreenPos();
  ImDrawList *draw_list = ImGui::GetWindowDrawList();
  const ImU32 border = ImGui::GetColorU32(color_rgb(92, 96, 98));
  const ImU32 text = ImGui::GetColorU32(ImGuiCol_Text);
  const ImU32 disabled = ImGui::GetColorU32(ImGuiCol_TextDisabled);

  push_mono_font();
  for (int bit = 0; bit < kBitColumns; ++bit) {
    char label[2] = {static_cast<char>('7' - bit), '\0'};
    const ImVec2 cell_min(origin.x + row_header_w + static_cast<float>(bit) * col_w, origin.y);
    const ImVec2 size = ImGui::CalcTextSize(label);
    draw_list->AddText(ImVec2(cell_min.x + (col_w - size.x) * 0.5f, cell_min.y + (row_h - size.y) * 0.5f), disabled, label);
  }
  const ImVec2 hex_min(origin.x + row_header_w + static_cast<float>(kBitColumns) * col_w, origin.y);
  const ImVec2 hex_size = ImGui::CalcTextSize("HEX");
  draw_list->AddText(ImVec2(hex_min.x + (col_w - hex_size.x) * 0.5f, hex_min.y + (row_h - hex_size.y) * 0.5f), disabled, "HEX");

  for (size_t row = 0; row < grid.rows.size(); ++row) {
    const float y = origin.y + row_h * static_cast<float>(row + 1);
    char row_label[32];
    std::snprintf(row_label, sizeof(row_label), "%zu", row);
    const ImVec2 row_label_size = ImGui::CalcTextSize(row_label);
    draw_list->AddText(ImVec2(origin.x + (row_header_w - row_label_size.x) * 0.5f, y + (row_h - row_label_size.y) * 0.5f),
                       disabled, row_label);

    for (int bit = 0; bit < kBitColumns; ++bit) {
      const BinaryBitCell &cell = grid.rows[row][static_cast<size_t>(bit)];
      const int bit_index = static_cast<int>(row * 8 + static_cast<size_t>(7 - bit));
      const ImVec2 cell_min(origin.x + row_header_w + static_cast<float>(bit) * col_w, y);
      const ImVec2 cell_max(cell_min.x + col_w, cell_min.y + row_h);
      draw_list->AddRectFilled(cell_min, cell_max, heat_color(cell.flip_count, grid.max_flip_count));
      if (drag.active && drag.id == id && drag.dragged) {
        const int first = std::min(drag.anchor_bit, drag.current_bit);
        const int last = std::max(drag.anchor_bit, drag.current_bit);
        if (bit_index >= first && bit_index <= last) {
          draw_list->AddRectFilled(cell_min, cell_max, ImGui::GetColorU32(color_rgb(82, 141, 255, 0.42f)));
        }
      }
      draw_list->AddRect(cell_min, cell_max, border);

      char value[2] = {static_cast<char>(cell.value ? '1' : '0'), '\0'};
      const ImVec2 value_size = ImGui::CalcTextSize(value);
      draw_list->AddText(ImVec2(cell_min.x + (col_w - value_size.x) * 0.5f, cell_min.y + (row_h - value_size.y) * 0.5f), text, value);

      ImGui::SetCursorScreenPos(cell_min);
      ImGui::PushID(static_cast<int>(row * 16 + static_cast<size_t>(bit)));
      ImGui::InvisibleButton("bit", ImVec2(col_w, row_h));
      if (!drag.active && ImGui::IsItemHovered() && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        drag.active = true;
        drag.dragged = false;
        drag.id = id;
        drag.anchor_bit = bit_index;
        drag.current_bit = bit_index;
        drag.status.clear();
      }
      if (drag.active && drag.id == id && ImGui::IsItemHovered()) {
        drag.current_bit = bit_index;
        if (drag.current_bit != drag.anchor_bit) drag.dragged = true;
      }
      if (ImGui::IsItemHovered()) {
        if (drag.active && drag.id == id && drag.dragged) {
          const int first = std::min(drag.anchor_bit, drag.current_bit);
          const int last = std::max(drag.anchor_bit, drag.current_bit);
          ImGui::SetTooltip("create signal bits %d-%d", first, last);
        } else {
          ImGui::SetTooltip("byte %zu bit %d\nvalue %u\nflips %u", row, 7 - bit, static_cast<unsigned>(cell.value), cell.flip_count);
        }
      }
      ImGui::PopID();
    }

    const ImVec2 hex_cell_min(origin.x + row_header_w + static_cast<float>(kBitColumns) * col_w, y);
    const ImVec2 hex_cell_max(hex_cell_min.x + col_w, hex_cell_min.y + row_h);
    draw_list->AddRectFilled(hex_cell_min, hex_cell_max, ImGui::GetColorU32(color_rgb(76, 80, 82)));
    draw_list->AddRect(hex_cell_min, hex_cell_max, border);
    char hex[3];
    std::snprintf(hex, sizeof(hex), "%02X", grid.latest_data[row]);
    const ImVec2 hex_text = ImGui::CalcTextSize(hex);
    draw_list->AddText(ImVec2(hex_cell_min.x + (col_w - hex_text.x) * 0.5f, hex_cell_min.y + (row_h - hex_text.y) * 0.5f), text, hex);
  }

  if (drag.active && drag.id == id && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
    if (drag.dragged) {
      Signal draft;
      std::string error;
      if (binary_signal_from_bit_range(drag.anchor_bit, drag.current_bit, &draft, &error) &&
          commit_signal_add(&session.dbc_undo(), dbc(), id, draft, static_cast<uint32_t>(grid.latest_data.size()), &error)) {
        drag.status = "Created DBC signal";
      } else {
        drag.status = error.empty() ? "Signal create failed" : error;
      }
    }
    drag.active = false;
    drag.dragged = false;
    drag.anchor_bit = -1;
    drag.current_bit = -1;
  }

  ImGui::SetCursorScreenPos(origin);
  ImGui::Dummy(ImVec2(total_w, total_h));
  pop_mono_font();
}

}  // namespace loggy
