#include "tools/loggy/panes/find_bits.h"

#include "tools/loggy/shell/workspace.h"

#include "imgui.h"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <optional>
#include <unordered_map>

namespace loggy {
namespace {

bool input_text_string(const char *label, std::string *value, size_t capacity) {
  if (value == nullptr) return false;
  std::vector<char> buffer(std::max(capacity, value->size() + 1), '\0');
  std::snprintf(buffer.data(), buffer.size(), "%s", value->c_str());
  if (!ImGui::InputText(label, buffer.data(), buffer.size())) return false;
  *value = buffer.data();
  return true;
}

uint8_t bit_value_at(const std::vector<uint8_t> &data, int byte_idx, int bit_idx) {
  if (byte_idx < 0 || bit_idx < 0 || bit_idx > 7 || byte_idx >= static_cast<int>(data.size())) return 0;
  return static_cast<uint8_t>((data[static_cast<size_t>(byte_idx)] >> (7 - bit_idx)) & 1U);
}

}  // namespace

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
  }).dump();
}

FindBitsParams find_bits_params_from_state(const FindBitsPaneState &state, TimeRange range) {
  FindBitsParams params;
  params.range = range;
  const MessageId source = MessageId::fromString(state.source);
  params.source_bus = source.source;
  params.source_address = source.address;
  params.byte_idx = std::clamp(state.byte_idx, 0, CAN_MAX_DATA_BYTES - 1);
  params.bit_idx = std::clamp(state.bit_idx, 0, 7);
  params.find_bus = static_cast<uint8_t>(std::clamp(state.find_bus, 0, 255));
  params.equal = state.equal;
  params.min_msgs = std::max(0, state.min_msgs);
  return params;
}

std::vector<FindBitsEvent> collect_find_bits_events(const Store &store, const FindBitsParams &params) {
  std::vector<FindBitsEvent> out;
  const MessageId source_id{.source = params.source_bus, .address = params.source_address};
  const CanEventView source = store.canEvents(source_id, params.range);
  if (source.events.empty()) return out;

  const std::vector<MessageId> ids = store.canMessageIds();
  std::vector<MessageId> targets;
  for (const MessageId &id : ids) {
    if (id.source == params.find_bus && id != source_id) targets.push_back(id);
  }
  std::sort(targets.begin(), targets.end());

  for (const CanEvent &source_event : source.events) {
    const uint8_t source_value = bit_value_at(source_event.data, params.byte_idx, params.bit_idx);
    for (const MessageId &id : targets) {
      const CanEventView target_view = store.canEvents(id, {params.range.start, source_event.mono_time});
      if (target_view.events.empty()) continue;
      const CanEvent &target_event = target_view.events.back();
      out.push_back({
        .mono_time = source_event.mono_time,
        .source_value = source_value,
        .id = id,
        .data = target_event.data,
      });
    }
  }
  return out;
}

std::vector<FindBitsRow> scan_find_bits_events(const std::vector<FindBitsEvent> &events,
                                               const FindBitsParams &params) {
  struct Accum {
    uint32_t total = 0;
    uint32_t mismatches = 0;
  };
  std::unordered_map<uint64_t, Accum> accum;
  auto key_for = [](uint32_t address, uint32_t byte_idx, uint32_t bit_idx) {
    return (static_cast<uint64_t>(address) << 16) | (static_cast<uint64_t>(byte_idx) << 8) | bit_idx;
  };

  for (const FindBitsEvent &event : events) {
    for (size_t byte_idx = 0; byte_idx < event.data.size(); ++byte_idx) {
      for (int bit_idx = 0; bit_idx < 8; ++bit_idx) {
        const uint8_t target_value = bit_value_at(event.data, static_cast<int>(byte_idx), bit_idx);
        const bool matched = params.equal ? target_value == event.source_value : target_value != event.source_value;
        Accum &row = accum[key_for(event.id.address, static_cast<uint32_t>(byte_idx), static_cast<uint32_t>(bit_idx))];
        ++row.total;
        if (!matched) ++row.mismatches;
      }
    }
  }

  std::vector<FindBitsRow> rows;
  rows.reserve(accum.size());
  for (const auto &[key, value] : accum) {
    if (static_cast<int>(value.total) <= params.min_msgs) continue;
    FindBitsRow row;
    row.address = static_cast<uint32_t>(key >> 16);
    row.byte_idx = static_cast<uint32_t>((key >> 8) & 0xFF);
    row.bit_idx = static_cast<uint32_t>(key & 0xFF);
    row.total = value.total;
    row.mismatches = value.mismatches;
    row.percent = value.total == 0 ? 0.0f : 100.0f * static_cast<float>(value.mismatches) / static_cast<float>(value.total);
    rows.push_back(row);
  }
  std::sort(rows.begin(), rows.end(), [](const FindBitsRow &a, const FindBitsRow &b) {
    return std::tie(a.percent, a.mismatches, a.address, a.byte_idx, a.bit_idx) <
           std::tie(b.percent, b.mismatches, b.address, b.byte_idx, b.bit_idx);
  });
  if (rows.size() > params.max_rows) rows.resize(params.max_rows);
  return rows;
}

FindBitsJob make_find_bits_job(const Store &store, const FindBitsParams &params) {
  FindBitsJob job;
  job.store = &store;
  job.params = params;
  job.ids = store.canMessageIds();
  job.ids.erase(std::remove_if(job.ids.begin(), job.ids.end(), [&](const MessageId &id) {
    return id.source != params.find_bus ||
           (id.source == params.source_bus && id.address == params.source_address);
  }), job.ids.end());
  std::sort(job.ids.begin(), job.ids.end());
  job.done = false;
  return job;
}

bool step_find_bits_job(FindBitsJob &job, size_t max_messages) {
  if (job.done || job.store == nullptr) return true;
  const MessageId source_id{.source = job.params.source_bus, .address = job.params.source_address};
  const CanEventView source = job.store->canEvents(source_id, job.params.range);
  if (source.events.empty()) {
    job.done = true;
    return true;
  }

  size_t visited = 0;
  while (visited < max_messages && job.id_index < job.ids.size()) {
    const MessageId id = job.ids[job.id_index++];
    ++visited;
    const CanEventView target = job.store->canEvents(id, job.params.range);
    if (target.events.empty()) continue;

    size_t target_idx = 0;
    for (const CanEvent &source_event : source.events) {
      while (target_idx + 1 < target.events.size() && target.events[target_idx + 1].mono_time <= source_event.mono_time) {
        ++target_idx;
      }
      if (target.events[target_idx].mono_time > source_event.mono_time) continue;
      job.events.push_back({
        .mono_time = source_event.mono_time,
        .source_value = bit_value_at(source_event.data, job.params.byte_idx, job.params.bit_idx),
        .id = id,
        .data = target.events[target_idx].data,
      });
    }
  }
  job.done = job.id_index >= job.ids.size();
  if (job.done) job.rows = scan_find_bits_events(job.events, job.params);
  return job.done;
}

void activate_find_bits_row(Session &session, std::string_view selection_group,
                            const FindBitsRow &row, uint8_t bus) {
  SelectionContext &selection = session.selection(selection_group);
  selection.selected_msg_id = MessageId{.source = bus, .address = row.address};
  selection.has_selected_msg = true;
}

void draw_find_bits_pane(Session &session, PaneInstance &pane) {
  static FindBitsJob job;
  static std::vector<FindBitsRow> rows;

  FindBitsPaneState state = parse_find_bits_pane_state(pane.state_json);
  bool changed = false;
  SelectionContext &selection = session.selection(pane.selection_group);
  if (selection.has_selected_msg && state.source == "0:47") {
    state.source = selection.selected_msg_id.toString();
    changed = true;
  }

  ImGui::SetNextItemWidth(92.0f);
  changed |= input_text_string("Source", &state.source, 32);
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

  if (ImGui::Button(job.done ? "Scan" : "Restart")) {
    job = make_find_bits_job(session.store(), find_bits_params_from_state(state, session.view_range().range()));
    rows.clear();
    state.status = "Scanning";
    changed = true;
  }
  if (!job.done && job.store == &session.store()) {
    if (step_find_bits_job(job, 64)) rows = job.rows;
    state.status = job.done ? ("Found " + std::to_string(rows.size()) + " bit matches") :
                              ("Scanning " + std::to_string(job.id_index) + "/" + std::to_string(job.ids.size()));
    changed = true;
  }
  ImGui::SameLine();
  ImGui::TextDisabled("%s", state.status.c_str());

  if (changed) pane.state_json = find_bits_pane_state_json(state);

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
