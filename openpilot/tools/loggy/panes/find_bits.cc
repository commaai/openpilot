#include "tools/loggy/panes/find_bits.h"

#include "tools/loggy/backend/scan.h"
#include "tools/loggy/backend/session.h"
#include "tools/loggy/backend/store.h"
#include "tools/loggy/shell/theme.h"
#include "tools/loggy/shell/workspace.h"

#include "imgui.h"
#include "json11/json11.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <any>
#include <string>
#include <string_view>
#include <optional>
#include <vector>
#include <algorithm>

namespace loggy {
namespace {

constexpr size_t kFindBitsHistoryLimit = 6;
constexpr size_t kFindBitsMaxRows = 512;

struct FindBitsPaneState {
  std::string source = "0:47";
  int byte_idx = 0;
  int bit_idx = 0;
  int find_bus = 0;
  bool equal = true;
  int min_msgs = 0;
  std::string status;
  int history_index = -1;
  std::vector<FindBitsParams> history;
};

struct FindBitsPaneTransientState {
  FindBitsPaneState state;
  std::string state_json;
  bool valid = false;
  FindBitsJob job;
  std::vector<FindBitsRow> rows;
};

FindBitsPaneState parse_find_bits_pane_state(std::string_view state_json);
std::string find_bits_pane_state_json(const FindBitsPaneState &state);
FindBitsParams find_bits_params_from_state(const FindBitsPaneState &state, TimeRange range);

FindBitsPaneTransientState &find_bits_transient_state(PaneInstance &pane) {
  if (FindBitsPaneTransientState *state = std::any_cast<FindBitsPaneTransientState>(&pane.transient_state)) return *state;
  pane.transient_state = FindBitsPaneTransientState{};
  return std::any_cast<FindBitsPaneTransientState &>(pane.transient_state);
}

FindBitsPaneState &find_bits_pane_state(PaneInstance &pane) {
  FindBitsPaneTransientState &transient = find_bits_transient_state(pane);
  if (!transient.valid || transient.state_json != pane.state_json) {
    transient.state = parse_find_bits_pane_state(pane.state_json);
    transient.state_json = pane.state_json;
    transient.valid = true;
    transient.job = FindBitsJob{};
    transient.rows.clear();
  }
  return transient.state;
}

std::string find_bits_history_entry_label(const FindBitsParams &params) {
  const MessageId source{.source = params.source_bus, .address = params.source_address};
  std::string out = source.to_string();
  out += " byte ";
  out += std::to_string(params.byte_idx);
  out += " bit ";
  out += std::to_string(params.bit_idx);
  out += " find ";
  out += std::to_string(params.find_bus);
  out += params.equal ? " ==" : " !=";
  out += " min ";
  out += std::to_string(params.min_msgs);
  return out;
}

void find_bits_apply_history_entry(FindBitsPaneState &state, const FindBitsParams &params) {
  const MessageId source{.source = params.source_bus, .address = params.source_address};
  state.source = source.to_string();
  state.byte_idx = params.byte_idx;
  state.bit_idx = params.bit_idx;
  state.find_bus = params.find_bus;
  state.equal = params.equal;
  state.min_msgs = params.min_msgs;
}

void find_bits_record_history_entry(FindBitsPaneState &state, size_t max_history = 6) {
  FindBitsParams params;
  params.source_bus = MessageId::from_string(state.source).source;
  params.source_address = MessageId::from_string(state.source).address;
  params.byte_idx = state.byte_idx;
  params.bit_idx = state.bit_idx;
  params.find_bus = static_cast<uint8_t>(std::clamp(state.find_bus, 0, 255));
  params.equal = state.equal;
  params.min_msgs = state.min_msgs;
  state.history.erase(std::remove_if(state.history.begin(), state.history.end(), [&](const FindBitsParams &entry) {
    return entry.source_bus == params.source_bus && entry.source_address == params.source_address &&
           entry.byte_idx == params.byte_idx && entry.bit_idx == params.bit_idx && entry.find_bus == params.find_bus &&
           entry.equal == params.equal && entry.min_msgs == params.min_msgs;
  }), state.history.end());
  state.history.push_back(params);
  while (state.history.size() > max_history) state.history.erase(state.history.begin());
  state.history_index = static_cast<int>(state.history.size()) - 1;
}

FindBitsPaneState parse_find_bits_pane_state(std::string_view state_json) {
  FindBitsPaneState state;
  std::string err;
  const json11::Json json = json11::Json::parse(std::string(state_json), err);
  if (!err.empty() || !json.is_object()) return state;
  if (json["source"].is_string()) state.source = json["source"].string_value();
  if (json["byte"].is_number()) state.byte_idx = std::clamp(json["byte"].int_value(), 0, CAN_MAX_DATA_BYTES - 1);
  if (json["bit"].is_number()) state.bit_idx = std::clamp(json["bit"].int_value(), 0, 7);
  if (json["find_bus"].is_number()) state.find_bus = std::clamp(json["find_bus"].int_value(), 0, 255);
  if (json["equal"].is_bool()) state.equal = json["equal"].bool_value();
  if (json["min_msgs"].is_number()) state.min_msgs = std::max(0, json["min_msgs"].int_value());
  if (json["status"].is_string()) state.status = json["status"].string_value();
  if (json["history_index"].is_number()) state.history_index = json["history_index"].int_value();
  if (json["history"].is_array()) {
    for (const json11::Json &item : json["history"].array_items()) {
      if (!item.is_object()) continue;
      FindBitsParams params;
      if (item["source"].is_string()) {
        const MessageId source = MessageId::from_string(item["source"].string_value());
        params.source_bus = source.source;
        params.source_address = source.address;
      }
      if (item["byte_idx"].is_number()) params.byte_idx = std::clamp(item["byte_idx"].int_value(), 0, CAN_MAX_DATA_BYTES - 1);
      if (item["bit_idx"].is_number()) params.bit_idx = std::clamp(item["bit_idx"].int_value(), 0, 7);
      if (item["find_bus"].is_number()) params.find_bus = static_cast<uint8_t>(std::clamp(item["find_bus"].int_value(), 0, 255));
      if (item["equal"].is_bool()) params.equal = item["equal"].bool_value();
      if (item["min_msgs"].is_number()) params.min_msgs = std::max(0, item["min_msgs"].int_value());
      state.history.push_back(params);
      while (state.history.size() > kFindBitsHistoryLimit) state.history.erase(state.history.begin());
    }
  }
  state.history_index = std::clamp(state.history_index, -1, static_cast<int>(state.history.size()) - 1);
  return state;
}

std::string find_bits_pane_state_json(const FindBitsPaneState &state) {
  return json11::Json(json11::Json::object{
    {"source", state.source},
    {"byte", state.byte_idx},
    {"bit", state.bit_idx},
    {"find_bus", state.find_bus},
    {"equal", state.equal},
    {"min_msgs", state.min_msgs},
    {"status", state.status},
    {"history_index", state.history_index},
    {"history", [&]() {
      json11::Json::array out;
      out.reserve(state.history.size());
      for (const FindBitsParams &params : state.history) {
        const MessageId source{.source = params.source_bus, .address = params.source_address};
        out.push_back(json11::Json::object{
          {"source", source.to_string()},
          {"byte_idx", params.byte_idx},
          {"bit_idx", params.bit_idx},
          {"find_bus", params.find_bus},
          {"equal", params.equal},
          {"min_msgs", params.min_msgs},
        });
      }
      return json11::Json(out);
    }()},
  }).dump();
}

FindBitsParams find_bits_params_from_state(const FindBitsPaneState &state, TimeRange range) {
  FindBitsParams params;
  params.max_rows = kFindBitsMaxRows;
  params.range = range;
  const MessageId source = MessageId::from_string(state.source);
  params.source_bus = source.source;
  params.source_address = source.address;
  params.byte_idx = std::clamp(state.byte_idx, 0, CAN_MAX_DATA_BYTES - 1);
  params.bit_idx = std::clamp(state.bit_idx, 0, 7);
  params.find_bus = static_cast<uint8_t>(std::clamp(state.find_bus, 0, 255));
  params.equal = state.equal;
  params.min_msgs = std::max(0, state.min_msgs);
  return params;
}

void activate_find_bits_row(Session &session, std::string_view selection_group,
                            const FindBitsRow &row, uint8_t bus) {
  SelectionContext &selection = session.selection(selection_group);
  selection.selected_msg_id = MessageId{.source = bus, .address = row.address};
  selection.has_selected_msg = true;
}

}  // namespace

void draw_find_bits_pane(Session &session, PaneInstance &pane) {
  FindBitsPaneTransientState &transient = find_bits_transient_state(pane);
  FindBitsJob &job = transient.job;
  std::vector<FindBitsRow> &rows = transient.rows;
  FindBitsPaneState &state = find_bits_pane_state(pane);
  bool changed = false;
  state.history_index = std::clamp(state.history_index, -1, static_cast<int>(state.history.size()) - 1);
  SelectionContext &selection = session.selection(pane.selection_group);
  if (selection.has_selected_msg && state.source == "0:47") {
    state.source = selection.selected_msg_id.to_string();
    changed = true;
  }

  ImGui::SetNextItemWidth(92.0f);
  changed |= input_text_with_hint("Source", "", &state.source);
  if (ImGui::GetContentRegionAvail().x > 112.0f) ImGui::SameLine();
  ImGui::SetNextItemWidth(64.0f);
  changed |= ImGui::InputInt("Byte", &state.byte_idx);
  if (ImGui::GetContentRegionAvail().x > 96.0f) ImGui::SameLine();
  ImGui::SetNextItemWidth(56.0f);
  changed |= ImGui::InputInt("Bit", &state.bit_idx);
  if (ImGui::GetContentRegionAvail().x > 112.0f) ImGui::SameLine();
  ImGui::SetNextItemWidth(64.0f);
  changed |= ImGui::InputInt("Bus", &state.find_bus);

  changed |= ImGui::Checkbox("Equal", &state.equal);
  if (ImGui::GetContentRegionAvail().x > 118.0f) ImGui::SameLine();
  ImGui::SetNextItemWidth(76.0f);
  changed |= ImGui::InputInt("Min", &state.min_msgs);

  if (!state.history.empty()) {
    const std::vector<std::string> history_labels = [&]() {
      std::vector<std::string> labels;
      labels.reserve(state.history.size());
      for (const FindBitsParams &params : state.history) labels.push_back(find_bits_history_entry_label(params));
      return labels;
    }();
    const std::vector<const char *> history_items = [&]() {
      std::vector<const char *> items;
      items.reserve(history_labels.size());
      for (const std::string &label : history_labels) items.push_back(label.c_str());
      return items;
    }();
    ImGui::SetNextItemWidth(220.0f);
    int history_index = std::max(state.history_index, 0);
    if (ImGui::Combo("History", &history_index, history_items.data(), static_cast<int>(history_items.size()))) {
      state.history_index = history_index;
      changed = true;
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("Apply")) {
      find_bits_apply_history_entry(state, state.history[static_cast<size_t>(std::clamp(state.history_index, 0, static_cast<int>(state.history.size()) - 1))]);
      changed = true;
    }
  }

  if (ImGui::Button(job.done ? "Scan" : "Restart")) {
    find_bits_record_history_entry(state);
    job = make_find_bits_job(session.store, find_bits_params_from_state(state, session.view_range.range()));
    rows.clear();
    state.status = "Scanning";
    changed = true;
  }
  if (!job.done && job.store == &session.store) {
    if (step_find_bits_job(job, 64)) rows = job.rows;
    state.status = job.done ? ("Found " + std::to_string(rows.size()) + " bit matches") :
                              ("Scanning " + std::to_string(job.id_index) + "/" + std::to_string(job.ids.size()));
    changed = true;
  }
  ImGui::SameLine();
  ImGui::TextDisabled("%s", state.status.c_str());

  if (changed) {
    pane.state_json = find_bits_pane_state_json(state);
    transient.state_json = pane.state_json;
  }

  if (ImGui::BeginTable("##find_bits_results", 6, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY,
                        ImGui::GetContentRegionAvail())) {
    ImGui::TableSetupColumn("Addr", ImGuiTableColumnFlags_WidthFixed, 72.0f);
    ImGui::TableSetupColumn("Byte", ImGuiTableColumnFlags_WidthFixed, 46.0f);
    ImGui::TableSetupColumn("Bit", ImGuiTableColumnFlags_WidthFixed, 38.0f);
    ImGui::TableSetupColumn("Miss", ImGuiTableColumnFlags_WidthFixed, 48.0f);
    ImGui::TableSetupColumn("Total", ImGuiTableColumnFlags_WidthFixed, 48.0f);
    ImGui::TableSetupColumn("%", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableHeadersRow();
    ImGuiListClipper clipper;
    clipper.Begin(static_cast<int>(rows.size()), std::max(ImGui::GetFrameHeight(), ImGui::GetTextLineHeight() + 8.0f));
    while (clipper.Step()) {
      for (int row_idx = clipper.DisplayStart; row_idx < clipper.DisplayEnd; ++row_idx) {
        const FindBitsRow &row = rows[static_cast<size_t>(row_idx)];
        ImGui::PushID(row_idx);
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        char addr[32];
        std::snprintf(addr, sizeof(addr), "0x%X", row.address);
        const bool clicked = ImGui::Selectable(addr, false, ImGuiSelectableFlags_SpanAllColumns);
        if (clicked || ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
          activate_find_bits_row(session, pane.selection_group, row, static_cast<uint8_t>(std::clamp(state.find_bus, 0, 255)));
        }
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%u", row.byte_idx);
        ImGui::TableSetColumnIndex(2);
        ImGui::Text("%u", row.bit_idx);
        ImGui::TableSetColumnIndex(3);
        ImGui::Text("%u", row.mismatches);
        ImGui::TableSetColumnIndex(4);
        ImGui::Text("%u", row.total);
        ImGui::TableSetColumnIndex(5);
        ImGui::Text("%.1f", row.percent);
        ImGui::PopID();
      }
    }
    ImGui::EndTable();
  }
}

}  // namespace loggy
