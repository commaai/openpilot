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
#include <string>
#include <vector>

namespace loggy {
namespace {

struct SignalEditCache {
  MessageId id;
  SignalEditModel edit;
  std::string val_desc_text;
  std::string val_desc_error;
  bool val_desc_valid = true;
  bool valid = false;
};

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

void load_signal_edit_cache(SignalEditCache *cache, const MessageId &id, const Signal &signal) {
  if (cache == nullptr) return;
  cache->id = id;
  cache->edit = signal_edit_model_from_signal(signal);
  cache->val_desc_text = signal_value_descriptions_text(signal.val_desc);
  cache->val_desc_error.clear();
  cache->val_desc_valid = true;
  cache->valid = true;
}

bool cache_matches_signal(const SignalEditCache &cache, const MessageId &id, const Signal &signal) {
  return cache.valid && cache.id == id && cache.edit.original_name == signal.name;
}

bool input_text_string(const char *label, std::string *value, size_t capacity) {
  if (value == nullptr) return false;
  std::vector<char> buffer(std::max<size_t>(capacity, value->size() + 1), '\0');
  std::snprintf(buffer.data(), buffer.size(), "%s", value->c_str());
  if (!ImGui::InputText(label, buffer.data(), buffer.size())) return false;
  *value = buffer.data();
  return true;
}

void draw_signal_color_swatch(const ColorRGBA &color) {
  const float size = ImGui::GetFrameHeight();
  const ImVec2 pos = ImGui::GetCursorScreenPos();
  ImGui::InvisibleButton("##signal_color", ImVec2(size, size));
  ImGui::GetWindowDrawList()->AddRectFilled(pos, ImVec2(pos.x + size, pos.y + size),
                                            IM_COL32(color.r, color.g, color.b, color.a), 3.0f);
  ImGui::GetWindowDrawList()->AddRect(pos, ImVec2(pos.x + size, pos.y + size),
                                      ImGui::GetColorU32(ImGuiCol_Border), 3.0f);
}

void draw_signal_sparkline(const SignalPaneRow &row) {
  constexpr float width = 92.0f;
  const float height = std::max(18.0f, ImGui::GetTextLineHeight() + 4.0f);
  const ImVec2 pos = ImGui::GetCursorScreenPos();
  ImGui::Dummy(ImVec2(width, height));

  ImDrawList *draw_list = ImGui::GetWindowDrawList();
  const ImVec2 max(pos.x + width, pos.y + height);
  draw_list->AddRectFilled(pos, max, ImGui::GetColorU32(color_rgb(48, 51, 53)), 2.0f);
  draw_list->AddRect(pos, max, ImGui::GetColorU32(color_rgb(82, 86, 88)), 2.0f);

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

void draw_signal_editor(Session &session, const MessageId &id, SignalPaneState *state,
                        SignalEditCache *cache, bool *changed) {
  if (state == nullptr || cache == nullptr || changed == nullptr || state->selected_signal.empty()) return;

  Msg *msg = dbc()->msg(id);
  Signal *signal = msg == nullptr ? nullptr : msg->sig(state->selected_signal);
  if (signal == nullptr) {
    state->selected_signal.clear();
    cache->valid = false;
    *changed = true;
    return;
  }
  if (!cache_matches_signal(*cache, id, *signal)) load_signal_edit_cache(cache, id, *signal);

  SignalEditModel &edit = cache->edit;
  ImGui::Separator();
  push_bold_font();
  ImGui::TextUnformatted("Signal");
  pop_bold_font();

  ImGui::SetNextItemWidth(180.0f);
  if (input_text_string("Name", &edit.name, 96)) *changed = true;
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
  if (input_text_string("Unit", &edit.unit, 64)) *changed = true;
  if (ImGui::GetContentRegionAvail().x > 164.0f) ImGui::SameLine();
  ImGui::SetNextItemWidth(160.0f);
  if (input_text_string("Receiver", &edit.receiver, 96)) *changed = true;
  ImGui::SetNextItemWidth(std::clamp(ImGui::GetContentRegionAvail().x * 0.55f, 220.0f, 520.0f));
  if (input_text_string("Comment", &edit.comment, 256)) *changed = true;

  ImGui::SetNextItemWidth(std::clamp(ImGui::GetContentRegionAvail().x * 0.55f, 220.0f, 620.0f));
  if (input_text_string("Value Table", &cache->val_desc_text, 1024)) {
    ValueDescription parsed;
    std::string parse_error;
    if (parse_signal_value_descriptions(cache->val_desc_text, &parsed, &parse_error)) {
      edit.val_desc = std::move(parsed);
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

  draw_signal_color_swatch(signal->color);
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("RGB %u, %u, %u", signal->color.r, signal->color.g, signal->color.b);
  }
  ImGui::SameLine();
  ImGui::TextDisabled("Precision %d", signal->precision);

  const bool has_edit = cache->val_desc_valid && signal_edit_model_changed(edit, *signal);
  if (!has_edit) ImGui::BeginDisabled();
  if (ImGui::Button("Apply")) {
    std::string error;
    if (!cache->val_desc_valid) {
      state->edit_error = cache->val_desc_error;
    } else if (apply_signal_edit(session.dbc_undo(), *dbc(), id, edit, &error)) {
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
    load_signal_edit_cache(cache, id, *signal);
    state->edit_error.clear();
    *changed = true;
  }

  if (ImGui::GetContentRegionAvail().x > 80.0f) ImGui::SameLine();
  if (ImGui::Button("Remove")) {
    std::string error;
    if (remove_signal_edit(session.dbc_undo(), *dbc(), id, state->selected_signal, &error)) {
      state->selected_signal.clear();
      state->edit_error.clear();
      cache->valid = false;
    } else {
      state->edit_error = error;
    }
    *changed = true;
  }

  if (ImGui::GetContentRegionAvail().x > 96.0f) ImGui::SameLine();
  UndoStack &undo = session.dbc_undo();
  const bool undo_disabled = !undo.canUndo();
  if (undo_disabled) ImGui::BeginDisabled();
  if (ImGui::Button("Undo")) {
    undo.undo();
    cache->valid = false;
    state->edit_error.clear();
    *changed = true;
  }
  if (undo_disabled) ImGui::EndDisabled();
  if (undo.canUndo() && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", undo.undoText().c_str());

  if (ImGui::GetContentRegionAvail().x > 96.0f) ImGui::SameLine();
  const bool redo_disabled = !undo.canRedo();
  if (redo_disabled) ImGui::BeginDisabled();
  if (ImGui::Button("Redo")) {
    undo.redo();
    cache->valid = false;
    state->edit_error.clear();
    *changed = true;
  }
  if (redo_disabled) ImGui::EndDisabled();
  if (undo.canRedo() && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", undo.redoText().c_str());

  if (!state->edit_error.empty()) {
    ImGui::SameLine();
    ImGui::TextDisabled("%s", state->edit_error.c_str());
  }
}

bool draw_signal_row(const SignalPaneRow &row, bool selected) {
  ImGui::TableNextRow(ImGuiTableRowFlags_None, std::max(ImGui::GetFrameHeight(), ImGui::GetTextLineHeight() + 8.0f));

  ImGui::TableSetColumnIndex(0);
  const bool clicked = ImGui::Selectable(row.name.c_str(), selected, ImGuiSelectableFlags_SpanAllColumns);

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
  draw_signal_sparkline(row);

  ImGui::TableSetColumnIndex(7);
  if (row.from_dbc) ImGui::TextUnformatted("--");
  else ImGui::Text("%u", row.flip_count);

  return clicked;
}

}  // namespace

void draw_signal_pane(Session &session, PaneInstance &pane) {
  static SignalEditCache edit_cache;
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
  if (ImGui::GetContentRegionAvail().x > 132.0f) ImGui::SameLine();
  ImGui::SetNextItemWidth(96.0f);
  if (ImGui::SliderInt("Spark", &state.sparkline_seconds, 1, 120, "%ds", ImGuiSliderFlags_AlwaysClamp)) changed = true;
  if (changed) pane.state_json = signal_pane_state_json(state);

  const std::vector<SignalPaneRow> rows = prepare_signal_pane_rows(session.store(), id, session.view_range().range(), state, dbc()->msg(id));
  const bool from_dbc = !rows.empty() && rows.front().from_dbc;
  if (ImGui::GetContentRegionAvail().x > 160.0f) ImGui::SameLine();
  ImGui::TextDisabled("ID %s | %zu %s", id.toString().c_str(), rows.size(), from_dbc ? "DBC signals" : "bit candidates");

  if (rows.empty()) {
    ImGui::TextDisabled("No signals or CAN bits in view");
    return;
  }

  if (from_dbc) {
    const auto selected_it = std::find_if(rows.begin(), rows.end(), [&](const SignalPaneRow &row) {
      return row.name == state.selected_signal;
    });
    if (selected_it == rows.end()) {
      state.selected_signal = rows.front().name;
      edit_cache.valid = false;
      changed = true;
    }
  }

  constexpr ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
                                    ImGuiTableFlags_BordersOuter | ImGuiTableFlags_SizingFixedFit |
                                    ImGuiTableFlags_ScrollY;
  ImVec2 table_size = ImGui::GetContentRegionAvail();
  if (from_dbc) table_size.y = std::clamp(table_size.y * 0.34f, 96.0f, 170.0f);
  if (!ImGui::BeginTable("##loggy_signal_table", 8, flags, table_size)) return;
  ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, 138.0f);
  ImGui::TableSetupColumn("Kind", ImGuiTableColumnFlags_WidthFixed, 58.0f);
  ImGui::TableSetupColumn("Start", ImGuiTableColumnFlags_WidthFixed, 48.0f);
  ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthFixed, 42.0f);
  ImGui::TableSetupColumn("Endian", ImGuiTableColumnFlags_WidthFixed, 54.0f);
  ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthFixed, 100.0f);
  ImGui::TableSetupColumn("Spark", ImGuiTableColumnFlags_WidthFixed, 100.0f);
  ImGui::TableSetupColumn("Flips", ImGuiTableColumnFlags_WidthFixed, 52.0f);
  ImGui::TableHeadersRow();

  ImGuiListClipper clipper;
  clipper.Begin(static_cast<int>(rows.size()), std::max(ImGui::GetFrameHeight(), ImGui::GetTextLineHeight() + 8.0f));
  while (clipper.Step()) {
    for (int row_idx = clipper.DisplayStart; row_idx < clipper.DisplayEnd; ++row_idx) {
      const SignalPaneRow &row = rows[static_cast<size_t>(row_idx)];
      if (draw_signal_row(row, row.from_dbc && row.name == state.selected_signal)) {
        state.selected_signal = row.from_dbc ? row.name : std::string();
        state.edit_error.clear();
        edit_cache.valid = false;
        changed = true;
      }
    }
  }
  ImGui::EndTable();
  if (from_dbc) {
    draw_signal_editor(session, id, &state, &edit_cache, &changed);
  }
  if (changed) pane.state_json = signal_pane_state_json(state);
}

}  // namespace loggy
