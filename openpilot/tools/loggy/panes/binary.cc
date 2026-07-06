#include "tools/loggy/panes/binary.h"

#include "tools/loggy/backend/csv.h"
#include "tools/loggy/backend/dbc/dbcmanager.h"
#include "tools/loggy/backend/session.h"
#include "tools/loggy/shell/theme.h"
#include "tools/loggy/shell/workspace.h"

#include "imgui.h"
#include "json11/json11.hpp"

#include <algorithm>
#include <array>
#include <any>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace loggy {
namespace {

struct BinaryPaneState {
  MessageId id = kDefaultLoggyMessageId;
  bool highlight_defined_bits = true;
  bool suppress_defined_bits = false;
};

BinaryPaneState parse_binary_pane_state(std::string_view state_json, std::optional<MessageId> selection = std::nullopt) {
  BinaryPaneState state;
  state.id = parse_message_id_state(state_json, selection);

  std::string err;
  const json11::Json json = json11::Json::parse(std::string(state_json), err);
  if (!err.empty() || !json.is_object()) return state;
  if (json["highlight_defined_bits"].is_bool()) state.highlight_defined_bits = json["highlight_defined_bits"].bool_value();
  if (json["suppress_defined_bits"].is_bool()) state.suppress_defined_bits = json["suppress_defined_bits"].bool_value();
  return state;
}

std::string binary_pane_state_json(const BinaryPaneState &state) {
  return json11::Json(json11::Json::object{
    {"id", state.id.to_string()},
    {"highlight_defined_bits", state.highlight_defined_bits},
    {"suppress_defined_bits", state.suppress_defined_bits},
  }).dump();
}

bool binary_bit_is_dbc_defined(const Msg *msg, int bit_index) {
  if (msg == nullptr || bit_index < 0) return false;
  const size_t byte_index = static_cast<size_t>(bit_index / 8);
  const int bit = bit_index % 8;
  if (byte_index >= msg->mask.size()) return false;
  return (msg->mask[byte_index] & (1U << bit)) != 0;
}

bool binary_signal_contains_bit(const Signal &signal, int bit_index) {
  if (bit_index < 0) return false;
  Signal copy = signal;
  copy.update();
  int byte_index = copy.msb / 8;
  int bits_left = copy.size;
  while (byte_index >= 0 && byte_index < CAN_MAX_DATA_BYTES && bits_left > 0) {
    const int lsb = static_cast<int>(copy.lsb / 8) == byte_index ? copy.lsb : byte_index * 8;
    const int msb = static_cast<int>(copy.msb / 8) == byte_index ? copy.msb : (byte_index + 1) * 8 - 1;
    if (bit_index >= lsb && bit_index <= msb) return true;
    bits_left -= msb - lsb + 1;
    byte_index = copy.is_little_endian ? byte_index - 1 : byte_index + 1;
  }
  return false;
}

const Signal *binary_signal_at_bit(const Msg *msg, int bit_index) {
  if (msg == nullptr || bit_index < 0) return nullptr;
  for (const Signal *signal : msg->signals()) {
    if (signal != nullptr && binary_signal_contains_bit(*signal, bit_index)) return signal;
  }
  return nullptr;
}

std::optional<Signal> binary_signal_from_bit_range(int anchor_bit, int current_bit, std::string &error) {
  if (anchor_bit < 0 || current_bit < 0 ||
      anchor_bit >= CAN_MAX_DATA_BYTES * 8 || current_bit >= CAN_MAX_DATA_BYTES * 8) {
    error = "bit range must be 0-511";
    return std::nullopt;
  }

  const int start_bit = std::min(anchor_bit, current_bit);
  const int size = std::abs(current_bit - anchor_bit) + 1;
  Signal draft;
  draft.start_bit = start_bit;
  draft.size = size;
  draft.is_little_endian = true;
  draft.is_signed = false;
  draft.factor = 1.0;
  draft.offset = 0.0;
  draft.min = 0.0;
  draft.max = std::pow(2.0, static_cast<double>(size)) - 1.0;
  draft.receiver_name = DEFAULT_NODE_NAME;
  draft.update();
  error.clear();
  return draft;
}

// Drag anchor/current bits come from the binary grid in "physical" bit-index order: byte i,
// column c (0=leftmost/MSB..7=rightmost/LSB) is bit_index = i*8 + (7-c). That is exactly the
// domain little-endian start_bit/lsb/msb already live in (update_msb_lsb: lsb=start_bit,
// msb=start_bit+size-1 — a contiguous, increasing range).
//
// Big-endian (Motorola) start_bit is the *physical* bit index of the MSB, but the signal's
// occupied bits are only contiguous/increasing in a different domain: flip_bit_pos (an
// involution: flip_bit_pos(flip_bit_pos(x)) == x) maps a physical bit index to its mirrored
// position within the same byte. update_msb_lsb defines, for BE: msb = start_bit, and
// lsb = flip_bit_pos(flip_bit_pos(start_bit) + size - 1); applying flip_bit_pos to both sides
// gives flip_bit_pos(lsb) = flip_bit_pos(start_bit) + size - 1 — i.e. in the flip_bit_pos'd
// domain, [flip_bit_pos(start_bit), flip_bit_pos(lsb)] is contiguous and increasing, the same
// shape as the little-endian case. So: transform physical bit indices with flip_bit_pos before
// doing the resize's min/max edge math for BE, then transform the result back with the same
// (self-inverse) function to get the new start_bit. For LE the transform is the identity, so one
// code path serves both endians.
int to_signal_bit_domain(bool is_little_endian, int physical_bit) {
  return is_little_endian ? physical_bit : flip_bit_pos(physical_bit);
}

std::optional<Signal> binary_resized_signal_from_bit_range(const Msg *msg, const Signal &origin,
                                                           int anchor_bit, int current_bit,
                                                           std::string &error) {
  if (msg == nullptr) {
    error = "no DBC message selected";
    return std::nullopt;
  }
  if (!binary_signal_contains_bit(origin, anchor_bit)) {
    error = "drag must start_ on the selected signal";
    return std::nullopt;
  }
  if (current_bit < 0 || current_bit >= static_cast<int>(msg->size * 8)) {
    error = "signal bit range must fit in the message";
    return std::nullopt;
  }

  const bool le = origin.is_little_endian;
  const int origin_first = to_signal_bit_domain(le, origin.start_bit);
  const int origin_last = origin_first + origin.size - 1;
  const int anchor = to_signal_bit_domain(le, anchor_bit);
  const int current = to_signal_bit_domain(le, current_bit);
  const int keep_edge = std::abs(anchor - origin_first) <= std::abs(anchor - origin_last) ?
      origin_last : origin_first;
  const int new_first = std::min(keep_edge, current);
  const int new_last = std::max(keep_edge, current);
  if (new_first < 0 || new_last >= static_cast<int>(msg->size * 8)) {
    error = "signal bit range must fit in the message";
    return std::nullopt;
  }

  Signal copy = origin;
  copy.start_bit = to_signal_bit_domain(le, new_first);
  copy.size = new_last - new_first + 1;
  copy.max = std::pow(2.0, static_cast<double>(copy.size)) - 1.0;
  copy.update();
  error.clear();
  return copy;
}

bool binary_signal_overlaps_dbc_bits(const Msg *msg, int start_bit, int size, std::string &error) {
  if (msg == nullptr || start_bit < 0 || size <= 0) return false;
  for (int bit = start_bit; bit < start_bit + size; ++bit) {
    if (binary_bit_is_dbc_defined(msg, bit)) {
      error = "selection overlaps existing DBC bits at " + std::to_string(bit);
      return true;
    }
  }
  error.clear();
  return false;
}

// candidate must already have start_bit/size/is_little_endian set (msb/lsb are recomputed here).
// [start_bit, start_bit+size) is only a valid physical bit range for little-endian signals; a
// Motorola candidate's occupied bits follow the same msb-anchored, per-byte walk as
// binary_signal_contains_bit, so this mirrors that walk instead of assuming linear layout.
bool binary_signal_overlaps_other_dbc_bits(const Msg *msg, const std::string &signal_name,
                                          Signal candidate, std::string &error) {
  if (msg == nullptr || candidate.size <= 0) return false;
  candidate.update();
  int byte_index = candidate.msb / 8;
  int bits_left = candidate.size;
  while (byte_index >= 0 && byte_index < CAN_MAX_DATA_BYTES && bits_left > 0) {
    const int lsb = (candidate.lsb / 8) == byte_index ? candidate.lsb : byte_index * 8;
    const int msb = (candidate.msb / 8) == byte_index ? candidate.msb : (byte_index + 1) * 8 - 1;
    for (int bit = lsb; bit <= msb; ++bit) {
      for (const Signal *signal : msg->signals()) {
        if (signal == nullptr || signal->name == signal_name) continue;
        if (binary_signal_contains_bit(*signal, bit)) {
          error = "selection overlaps signal " + signal->name + " at bit " + std::to_string(bit);
          return true;
        }
      }
    }
    bits_left -= msb - lsb + 1;
    byte_index = candidate.is_little_endian ? byte_index - 1 : byte_index + 1;
  }
  error.clear();
  return false;
}

ImU32 heat_color(uint32_t flips, uint32_t max_flips) {
  if (flips == 0 || max_flips == 0) return ImGui::GetColorU32(binary_grid_background_color());
  const float alpha = 0.18f + 0.58f * (std::log2(1.0f + static_cast<float>(flips)) /
                                       std::log2(1.0f + static_cast<float>(max_flips)));
  return ImGui::GetColorU32(color_rgb(47, 101, 202, alpha));
}

struct BinaryDragState {
  bool active = false;
  bool dragged = false;
  MessageId id;
  int anchor_bit = -1;
  int current_bit = -1;
  bool resizing = false;
  std::string signal_name;
  std::string status;
};

BinaryDragState &binary_drag_state(PaneInstance &pane) {
  if (BinaryDragState *state = std::any_cast<BinaryDragState>(&pane.transient_state)) return *state;
  pane.transient_state = BinaryDragState{};
  return std::any_cast<BinaryDragState &>(pane.transient_state);
}

}  // namespace

void draw_binary_pane(Session &session, PaneInstance &pane) {
  SelectionContext &selection = session.selection(pane.selection_group);
  const std::optional<MessageId> selected = selection.has_selected_msg ? std::optional<MessageId>(selection.selected_msg_id) : std::nullopt;
  BinaryPaneState state = parse_binary_pane_state(pane.state_json, selected);
  const MessageId id = state.id;
  const std::optional<BinaryGrid> maybe_grid = build_binary_grid(session.store, id, session.view_range.range());
  Msg *dbc_msg = session.dbc.msg(id);

  ImGui::TextDisabled("ID %s", id.to_string().c_str());
  if (!maybe_grid.has_value()) {
    ImGui::TextDisabled("No CAN events in view");
    return;
  }

  const BinaryGrid &grid = *maybe_grid;
  ImGui::SameLine();
  ImGui::TextDisabled("| %zu events | latest %.3fs", grid.event_count, grid.last_time);
  BinaryDragState &drag = binary_drag_state(pane);
  if (!drag.status.empty()) {
    ImGui::SameLine();
    ImGui::TextDisabled("%s", drag.status.c_str());
  }
  if (dbc_msg != nullptr && !dbc_msg->mask.empty()) {
    bool changed = false;
    if (ImGui::GetContentRegionAvail().x > 126.0f) ImGui::SameLine();
    changed |= ImGui::Checkbox("Defined", &state.highlight_defined_bits);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Highlight bits covered by loaded DBC signals");
    if (ImGui::GetContentRegionAvail().x > 132.0f) ImGui::SameLine();
    changed |= ImGui::Checkbox("Suppress", &state.suppress_defined_bits);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Mute DBC-defined bits while inspecting unknown bits");
    if (changed) pane.state_json = binary_pane_state_json(state);
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
  draw_list->AddText(ImVec2(hex_min.x + (col_w - hex_size.x) * 0.5f, hex_min.y + (row_h - hex_size.y) * 0.5f),
                    disabled, "HEX");

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
      const bool defined_bit = binary_bit_is_dbc_defined(dbc_msg, bit_index);
      const bool suppressed_bit = state.suppress_defined_bits && defined_bit;
      const ImVec2 cell_min(origin.x + row_header_w + static_cast<float>(bit) * col_w, y);
      const ImVec2 cell_max(cell_min.x + col_w, cell_min.y + row_h);
      draw_list->AddRectFilled(cell_min, cell_max,
                               suppressed_bit ? ImGui::GetColorU32(color_rgb(45, 48, 50))
                                              : heat_color(cell.flip_count, grid.max_flip_count));
      if (state.highlight_defined_bits && defined_bit && !suppressed_bit) {
        draw_list->AddRectFilled(cell_min, cell_max, ImGui::GetColorU32(color_rgb(69, 126, 86, 0.32f)));
      }
      if (drag.active && drag.id == id && drag.dragged) {
        const int first = std::min(drag.anchor_bit, drag.current_bit);
        const int last = std::max(drag.anchor_bit, drag.current_bit);
        if (bit_index >= first && bit_index <= last) {
          draw_list->AddRectFilled(cell_min, cell_max, ImGui::GetColorU32(color_rgb(82, 141, 255, 0.42f)));
        }
      }
      draw_list->AddRect(cell_min, cell_max, border);
      if (state.highlight_defined_bits && defined_bit) {
        draw_list->AddRect(ImVec2(cell_min.x + 1.0f, cell_min.y + 1.0f),
                           ImVec2(cell_max.x - 1.0f, cell_max.y - 1.0f),
                           ImGui::GetColorU32(color_rgb(114, 175, 124)));
      }

      char value[2] = {static_cast<char>(cell.value ? '1' : '0'), '\0'};
      if (!suppressed_bit) {
        const ImVec2 value_size = ImGui::CalcTextSize(value);
        draw_list->AddText(ImVec2(cell_min.x + (col_w - value_size.x) * 0.5f, cell_min.y + (row_h - value_size.y) * 0.5f),
                          text, value);
      }

      ImGui::SetCursorScreenPos(cell_min);
      ImGui::PushID(static_cast<int>(row * 16 + static_cast<size_t>(bit)));
      ImGui::InvisibleButton("bit", ImVec2(col_w, row_h));
      if (!drag.active && !suppressed_bit && ImGui::IsItemHovered() && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        const Signal *resize_signal = binary_signal_at_bit(dbc_msg, bit_index);
        drag.active = true;
        drag.dragged = false;
        drag.id = id;
        drag.anchor_bit = bit_index;
        drag.current_bit = bit_index;
        drag.resizing = resize_signal != nullptr;
        drag.signal_name = resize_signal != nullptr ? resize_signal->name : std::string();
        drag.status.clear();
      }
      // Once the press on the anchor cell makes it ImGui's "active" item, IsItemHovered() on any
      // *other* cell reports false by default (ImGui only lets the active item claim hover unless
      // asked otherwise) -- so a per-cell IsItemHovered() loop can only ever see the drag land back
      // on the exact cell it started from. IsMouseHoveringRect is a plain geometric hit test with
      // no such exclusivity, so it tracks the true cursor position across every cell while the
      // button is held.
      if (drag.active && drag.id == id && ImGui::IsMouseHoveringRect(cell_min, cell_max)) {
        drag.current_bit = bit_index;
        if (drag.current_bit != drag.anchor_bit) drag.dragged = true;
      }
      if (ImGui::IsItemHovered() || (drag.active && drag.id == id && ImGui::IsMouseHoveringRect(cell_min, cell_max))) {
        if (drag.active && drag.id == id && drag.dragged) {
          const int first = std::min(drag.anchor_bit, drag.current_bit);
          const int last = std::max(drag.anchor_bit, drag.current_bit);
          if (drag.resizing) {
            ImGui::SetTooltip("resize %s with bits %d-%d", drag.signal_name.c_str(), first, last);
          } else {
            ImGui::SetTooltip("create signal bits %d-%d", first, last);
          }
        } else if (suppressed_bit) {
          ImGui::SetTooltip("byte %zu bit %d\nDBC-defined bit suppressed", row, 7 - bit);
        } else if (defined_bit) {
          const Signal *hover_signal = binary_signal_at_bit(dbc_msg, bit_index);
          if (hover_signal != nullptr) {
            ImGui::SetTooltip("byte %zu bit %d\nvalue %u\nflips %u\nDBC-defined bit\n%s",
                              row, 7 - bit, static_cast<unsigned>(cell.value), cell.flip_count,
                              hover_signal->name.c_str());
          } else {
            ImGui::SetTooltip("byte %zu bit %d\nvalue %u\nflips %u\nDBC-defined bit", row, 7 - bit,
                              static_cast<unsigned>(cell.value), cell.flip_count);
          }
        } else {
          ImGui::SetTooltip("byte %zu bit %d\nvalue %u\nflips %u", row, 7 - bit, static_cast<unsigned>(cell.value), cell.flip_count);
        }
      }
      ImGui::PopID();
    }

    const ImVec2 hex_cell_min(origin.x + row_header_w + static_cast<float>(kBitColumns) * col_w, y);
    const ImVec2 hex_cell_max(hex_cell_min.x + col_w, hex_cell_min.y + row_h);
    draw_list->AddRectFilled(hex_cell_min, hex_cell_max, ImGui::GetColorU32(binary_grid_background_color()));
    draw_list->AddRect(hex_cell_min, hex_cell_max, border);
    char hex[3];
    std::snprintf(hex, sizeof(hex), "%02X", grid.latest_data[row]);
    const ImVec2 hex_text = ImGui::CalcTextSize(hex);
    draw_list->AddText(ImVec2(hex_cell_min.x + (col_w - hex_text.x) * 0.5f, hex_cell_min.y + (row_h - hex_text.y) * 0.5f),
                      text, hex);
  }

  if (drag.active && drag.id == id && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
    if (drag.dragged) {
      std::string error;
      if (drag.resizing) {
        const Signal *origin_signal = dbc_msg != nullptr ? dbc_msg->sig(drag.signal_name) : nullptr;
        std::optional<Signal> edited =
            origin_signal != nullptr
                ? binary_resized_signal_from_bit_range(dbc_msg, *origin_signal, drag.anchor_bit,
                                                       drag.current_bit, error)
                : std::nullopt;
        if (origin_signal != nullptr &&
            edited.has_value() &&
            !binary_signal_overlaps_other_dbc_bits(dbc_msg, origin_signal->name, *edited, error) &&
            commit_signal_edit(session.dbc_undo, session.dbc, id, *origin_signal, *edited, error)) {
          drag.status = "Resized " + drag.signal_name;
        } else {
          drag.status = error.empty() ? "Signal resize failed" : error;
        }
      } else {
        std::optional<Signal> draft = binary_signal_from_bit_range(drag.anchor_bit, drag.current_bit, error);
        if (draft.has_value() &&
            !binary_signal_overlaps_dbc_bits(dbc_msg, draft->start_bit, draft->size, error) &&
            commit_signal_add(session.dbc_undo, session.dbc, id, *draft, static_cast<uint32_t>(grid.latest_data.size()), error)) {
          drag.status = "Created DBC signal";
        } else {
          drag.status = error.empty() ? "Signal create failed" : error;
        }
      }
    }
    drag.active = false;
    drag.dragged = false;
    drag.anchor_bit = -1;
    drag.current_bit = -1;
    drag.resizing = false;
    drag.signal_name.clear();
  }

  ImGui::SetCursorScreenPos(origin);
  ImGui::Dummy(ImVec2(total_w, total_h));
  pop_mono_font();
}

}  // namespace loggy
