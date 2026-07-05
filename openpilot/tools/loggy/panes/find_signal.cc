#include "tools/loggy/panes/find_signal.h"

#include "tools/loggy/shell/theme.h"
#include "tools/loggy/shell/workspace.h"

#include "imgui.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <optional>

namespace loggy {
namespace {

bool vector_contains_int(const std::vector<int> &values, int value) {
  return values.empty() || std::find(values.begin(), values.end(), value) != values.end();
}

bool vector_contains_address(const std::vector<uint32_t> &values, uint32_t value) {
  return values.empty() || std::find(values.begin(), values.end(), value) != values.end();
}

std::optional<uint32_t> parse_hex_address(std::string_view text) {
  std::string cleaned;
  for (char ch : text) {
    if (!std::isspace(static_cast<unsigned char>(ch))) cleaned += ch;
  }
  if (cleaned.empty()) return std::nullopt;
  char *end = nullptr;
  const unsigned long value = std::strtoul(cleaned.c_str(), &end, 16);
  if (end == cleaned.c_str() || *end != '\0' || value > std::numeric_limits<uint32_t>::max()) return std::nullopt;
  return static_cast<uint32_t>(value);
}

bool input_text_string(const char *label, std::string *value, size_t capacity) {
  if (value == nullptr) return false;
  std::vector<char> buffer(std::max(capacity, value->size() + 1), '\0');
  std::snprintf(buffer.data(), buffer.size(), "%s", value->c_str());
  if (!ImGui::InputText(label, buffer.data(), buffer.size())) return false;
  *value = buffer.data();
  return true;
}

Signal find_signal_candidate(int start_bit, int size, const FindSignalParams &params) {
  Signal signal;
  signal.start_bit = start_bit;
  signal.size = size;
  signal.is_little_endian = params.little_endian;
  signal.is_signed = params.is_signed;
  signal.factor = params.factor;
  signal.offset = params.offset;
  signal.min = 0.0;
  signal.max = std::pow(2.0, static_cast<double>(size)) - 1.0;
  signal.receiver_name = DEFAULT_NODE_NAME;
  signal.update();
  return signal;
}

uint32_t event_msg_size(const CanEventView &view) {
  size_t size = 0;
  for (const CanEvent &event : view.events) size = std::max(size, event.data.size());
  return static_cast<uint32_t>(std::min<size_t>(size, CAN_MAX_DATA_BYTES));
}

void advance_find_signal_cursor(FindSignalJob &job, int msg_bits) {
  ++job.start_bit;
  if (job.start_bit + job.size <= msg_bits) return;
  job.start_bit = 0;
  ++job.size;
  if (job.size <= job.params.max_size) return;
  ++job.id_index;
  job.size = job.params.min_size;
  job.start_bit = 0;
}

}  // namespace

const char *find_signal_compare_token(FindSignalCompare compare) {
  switch (compare) {
    case FindSignalCompare::Equal: return "eq";
    case FindSignalCompare::NotEqual: return "ne";
    case FindSignalCompare::Greater: return "gt";
    case FindSignalCompare::GreaterEqual: return "ge";
    case FindSignalCompare::Less: return "lt";
    case FindSignalCompare::LessEqual: return "le";
    case FindSignalCompare::Any:
    default: return "any";
  }
}

const char *find_signal_compare_label(FindSignalCompare compare) {
  switch (compare) {
    case FindSignalCompare::Equal: return "=";
    case FindSignalCompare::NotEqual: return "!=";
    case FindSignalCompare::Greater: return ">";
    case FindSignalCompare::GreaterEqual: return ">=";
    case FindSignalCompare::Less: return "<";
    case FindSignalCompare::LessEqual: return "<=";
    case FindSignalCompare::Any:
    default: return "Any";
  }
}

FindSignalCompare find_signal_compare_from_token(std::string_view token) {
  if (token == "eq" || token == "=") return FindSignalCompare::Equal;
  if (token == "ne" || token == "!=") return FindSignalCompare::NotEqual;
  if (token == "gt" || token == ">") return FindSignalCompare::Greater;
  if (token == "ge" || token == ">=") return FindSignalCompare::GreaterEqual;
  if (token == "lt" || token == "<") return FindSignalCompare::Less;
  if (token == "le" || token == "<=") return FindSignalCompare::LessEqual;
  return FindSignalCompare::Any;
}

bool find_signal_compare_value(double value, FindSignalCompare compare, double target) {
  if (!std::isfinite(value)) return false;
  switch (compare) {
    case FindSignalCompare::Equal: return std::abs(value - target) <= 1.0e-9;
    case FindSignalCompare::NotEqual: return std::abs(value - target) > 1.0e-9;
    case FindSignalCompare::Greater: return value > target;
    case FindSignalCompare::GreaterEqual: return value >= target;
    case FindSignalCompare::Less: return value < target;
    case FindSignalCompare::LessEqual: return value <= target;
    case FindSignalCompare::Any:
    default: return true;
  }
}

FindSignalPaneState parse_find_signal_pane_state(std::string_view state_json) {
  FindSignalPaneState state;
  std::string err;
  const json11::Json json = json11::Json::parse(std::string(state_json), err);
  if (!err.empty() || !json.is_object()) return state;
  if (json["bus"].is_number()) state.bus = std::clamp(json["bus"].int_value(), -1, 255);
  if (json["address"].is_string()) state.address_hex = json["address"].string_value();
  if (json["min_size"].is_number()) state.min_size = std::clamp(json["min_size"].int_value(), 1, 64);
  if (json["max_size"].is_number()) state.max_size = std::clamp(json["max_size"].int_value(), 1, 64);
  if (json["little_endian"].is_bool()) state.little_endian = json["little_endian"].bool_value();
  if (json["signed"].is_bool()) state.is_signed = json["signed"].bool_value();
  if (json["factor"].is_number()) state.factor = json["factor"].number_value();
  if (json["offset"].is_number()) state.offset = json["offset"].number_value();
  if (json["target"].is_number()) state.target_value = json["target"].number_value();
  if (json["compare"].is_string()) state.compare = find_signal_compare_from_token(json["compare"].string_value());
  if (json["status"].is_string()) state.status = json["status"].string_value();
  if (state.max_size < state.min_size) state.max_size = state.min_size;
  if (!std::isfinite(state.factor) || state.factor == 0.0) state.factor = 1.0;
  if (!std::isfinite(state.offset)) state.offset = 0.0;
  if (!std::isfinite(state.target_value)) state.target_value = 0.0;
  return state;
}

std::string find_signal_pane_state_json(const FindSignalPaneState &state) {
  return json11::Json(json11::Json::object{
    {"bus", state.bus},
    {"address", state.address_hex},
    {"min_size", state.min_size},
    {"max_size", state.max_size},
    {"little_endian", state.little_endian},
    {"signed", state.is_signed},
    {"factor", state.factor},
    {"offset", state.offset},
    {"target", state.target_value},
    {"compare", find_signal_compare_token(state.compare)},
    {"status", state.status},
  }).dump();
}

FindSignalParams find_signal_params_from_state(const FindSignalPaneState &state, TimeRange range) {
  FindSignalParams params;
  params.range = range;
  if (state.bus >= 0) params.buses.push_back(state.bus);
  if (const std::optional<uint32_t> address = parse_hex_address(state.address_hex)) params.addresses.push_back(*address);
  params.min_size = std::clamp(state.min_size, 1, CAN_MAX_DATA_BYTES * 8);
  params.max_size = std::clamp(std::max(state.max_size, state.min_size), params.min_size, CAN_MAX_DATA_BYTES * 8);
  params.little_endian = state.little_endian;
  params.is_signed = state.is_signed;
  params.factor = state.factor;
  params.offset = state.offset;
  params.target_value = state.target_value;
  params.compare = state.compare;
  return params;
}

FindSignalJob make_find_signal_job(const Store &store, const FindSignalParams &params) {
  FindSignalJob job;
  job.store = &store;
  job.params = params;
  job.params.min_size = std::clamp(job.params.min_size, 1, CAN_MAX_DATA_BYTES * 8);
  job.params.max_size = std::clamp(std::max(job.params.max_size, job.params.min_size),
                                   job.params.min_size, CAN_MAX_DATA_BYTES * 8);
  job.size = job.params.min_size;
  job.done = false;
  job.ids = store.canMessageIds();
  job.ids.erase(std::remove_if(job.ids.begin(), job.ids.end(), [&](const MessageId &id) {
    return !vector_contains_int(job.params.buses, id.source) ||
           !vector_contains_address(job.params.addresses, id.address);
  }), job.ids.end());
  std::sort(job.ids.begin(), job.ids.end());
  if (job.ids.empty()) job.done = true;
  return job;
}

bool step_find_signal_job(FindSignalJob &job, size_t max_candidates) {
  if (job.done || job.store == nullptr) return true;
  size_t visited = 0;
  while (visited < max_candidates && job.id_index < job.ids.size() && job.results.size() < job.params.max_results) {
    const MessageId id = job.ids[job.id_index];
    const CanEventView view = job.store->canEvents(id, job.params.range);
    const uint32_t msg_size = event_msg_size(view);
    const int msg_bits = static_cast<int>(msg_size * 8);
    if (view.events.empty() || msg_bits <= 0 || job.size > msg_bits) {
      ++job.id_index;
      job.size = job.params.min_size;
      job.start_bit = 0;
      continue;
    }

    if (job.start_bit + job.size > msg_bits) {
      advance_find_signal_cursor(job, msg_bits);
      continue;
    }

    ++visited;
    Signal signal = find_signal_candidate(job.start_bit, job.size, job.params);
    FindSignalResult result;
    result.id = id;
    result.sig = signal;
    result.msg_size = msg_size;
    for (const CanEvent &event : view.events) {
      if (event.data.size() < msg_size) continue;
      double value = 0.0;
      if (!signal.getValue(event.data.data(), event.data.size(), &value)) continue;
      if (!find_signal_compare_value(value, job.params.compare, job.params.target_value)) continue;
      if (result.matches.empty()) result.mono_time = event.mono_time;
      result.matches.push_back({event.mono_time, value});
    }
    if (!result.matches.empty()) job.results.push_back(std::move(result));
    advance_find_signal_cursor(job, msg_bits);
  }
  job.done = job.id_index >= job.ids.size() || job.results.size() >= job.params.max_results;
  return job.done;
}

std::vector<FindSignalResult> prepare_find_signal_candidates(const Store &store,
                                                             const FindSignalParams &params) {
  FindSignalJob job = make_find_signal_job(store, params);
  while (!step_find_signal_job(job, 2048)) {
  }
  return std::move(job.results);
}

bool commit_find_signal_result(Session &session, std::string_view selection_group,
                               const FindSignalResult &result, std::string *error) {
  if (result.msg_size == 0) {
    if (error != nullptr) *error = "candidate has no message size";
    return false;
  }
  Signal signal = result.sig;
  signal.name.clear();
  if (!commit_signal_add(&session.dbc_undo(), dbc(), result.id, std::move(signal), result.msg_size, error)) return false;
  SelectionContext &selection = session.selection(selection_group);
  selection.selected_msg_id = result.id;
  selection.has_selected_msg = true;
  return true;
}

void draw_find_signal_pane(Session &session, PaneInstance &pane) {
  static FindSignalJob job;
  static std::vector<FindSignalResult> results;

  FindSignalPaneState state = parse_find_signal_pane_state(pane.state_json);
  bool changed = false;

  ImGui::SetNextItemWidth(70.0f);
  changed |= ImGui::InputInt("Bus", &state.bus);
  if (ImGui::GetContentRegionAvail().x > 120.0f) ImGui::SameLine();
  ImGui::SetNextItemWidth(90.0f);
  changed |= input_text_string("Addr", &state.address_hex, 32);
  if (ImGui::GetContentRegionAvail().x > 120.0f) ImGui::SameLine();
  ImGui::SetNextItemWidth(66.0f);
  changed |= ImGui::InputInt("Min", &state.min_size);
  if (ImGui::GetContentRegionAvail().x > 120.0f) ImGui::SameLine();
  ImGui::SetNextItemWidth(66.0f);
  changed |= ImGui::InputInt("Max", &state.max_size);

  changed |= ImGui::Checkbox("Little Endian", &state.little_endian);
  if (ImGui::GetContentRegionAvail().x > 88.0f) ImGui::SameLine();
  changed |= ImGui::Checkbox("Signed", &state.is_signed);
  if (ImGui::GetContentRegionAvail().x > 118.0f) ImGui::SameLine();
  int compare_index = static_cast<int>(state.compare);
  ImGui::SetNextItemWidth(82.0f);
  if (ImGui::Combo("Match", &compare_index, "Any\0=\0!=\0>\0>=\0<\0<=\0")) {
    state.compare = static_cast<FindSignalCompare>(std::clamp(compare_index, 0, 6));
    changed = true;
  }
  if (ImGui::GetContentRegionAvail().x > 128.0f) ImGui::SameLine();
  ImGui::SetNextItemWidth(96.0f);
  changed |= ImGui::InputDouble("Value", &state.target_value, 0.0, 0.0, "%.9g");

  ImGui::SetNextItemWidth(96.0f);
  changed |= ImGui::InputDouble("Factor", &state.factor, 0.0, 0.0, "%.9g");
  if (ImGui::GetContentRegionAvail().x > 132.0f) ImGui::SameLine();
  ImGui::SetNextItemWidth(96.0f);
  changed |= ImGui::InputDouble("Offset", &state.offset, 0.0, 0.0, "%.9g");

  if (ImGui::Button(job.done ? "Search" : "Restart")) {
    job = make_find_signal_job(session.store(), find_signal_params_from_state(state, session.view_range().range()));
    results.clear();
    state.status = "Scanning";
    changed = true;
  }
  if (!job.done && job.store == &session.store()) {
    step_find_signal_job(job, 512);
    results = job.results;
    state.status = job.done ? ("Found " + std::to_string(results.size()) + " candidates") :
                              ("Scanning " + std::to_string(results.size()) + " candidates");
    changed = true;
  }
  ImGui::SameLine();
  ImGui::TextDisabled("%s", state.status.c_str());

  if (changed) pane.state_json = find_signal_pane_state_json(state);

  if (ImGui::BeginTable("##find_signal_results", 7, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY,
                        ImGui::GetContentRegionAvail())) {
    ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, 74.0f);
    ImGui::TableSetupColumn("Start", ImGuiTableColumnFlags_WidthFixed, 48.0f);
    ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthFixed, 42.0f);
    ImGui::TableSetupColumn("Endian", ImGuiTableColumnFlags_WidthFixed, 48.0f);
    ImGui::TableSetupColumn("Hits", ImGuiTableColumnFlags_WidthFixed, 50.0f);
    ImGui::TableSetupColumn("First", ImGuiTableColumnFlags_WidthFixed, 72.0f);
    ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableHeadersRow();
    ImGuiListClipper clipper;
    clipper.Begin(static_cast<int>(results.size()), std::max(ImGui::GetFrameHeight(), ImGui::GetTextLineHeight() + 8.0f));
    while (clipper.Step()) {
      for (int row_idx = clipper.DisplayStart; row_idx < clipper.DisplayEnd; ++row_idx) {
        const FindSignalResult &result = results[static_cast<size_t>(row_idx)];
        ImGui::PushID(row_idx);
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted(result.id.toString().c_str());
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%d", result.sig.start_bit);
        ImGui::TableSetColumnIndex(2);
        ImGui::Text("%d", result.sig.size);
        ImGui::TableSetColumnIndex(3);
        ImGui::TextUnformatted(result.sig.is_little_endian ? "LE" : "BE");
        ImGui::TableSetColumnIndex(4);
        ImGui::Text("%zu", result.matches.size());
        ImGui::TableSetColumnIndex(5);
        ImGui::Text("%.3f", result.mono_time);
        ImGui::TableSetColumnIndex(6);
        if (ImGui::SmallButton("Select")) {
          SelectionContext &selection = session.selection(pane.selection_group);
          selection.selected_msg_id = result.id;
          selection.has_selected_msg = true;
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Create")) {
          std::string error;
          state.status = commit_find_signal_result(session, pane.selection_group, result, &error)
                           ? "Created DBC signal"
                           : error;
          pane.state_json = find_signal_pane_state_json(state);
        }
        ImGui::PopID();
      }
    }
    ImGui::EndTable();
  }
}

}  // namespace loggy
