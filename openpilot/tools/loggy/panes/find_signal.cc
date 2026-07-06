#include "tools/loggy/panes/find_signal.h"

#include "tools/loggy/backend/scan.h"
#include "tools/loggy/backend/session.h"
#include "tools/loggy/backend/store.h"
#include "tools/loggy/backend/dbc/undo.h"
#include "tools/loggy/shell/theme.h"
#include "tools/loggy/shell/workspace.h"

#include "imgui.h"
#include "json11/json11.hpp"

#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <any>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include <algorithm>
#include <utility>

namespace loggy {
namespace {

struct FindSignalHistoryEntry {
  int bus = -1;
  std::string address_hex;
  int min_size = 1;
  int max_size = 12;
  bool little_endian = true;
  bool is_signed = false;
  double factor = 1.0;
  double offset = 0.0;
  double target_value = 0.0;
  FindSignalCompare compare = FindSignalCompare::Equal;
};

struct FindSignalPaneState {
  int bus = -1;
  std::string address_hex;
  std::string signal_name;
  int min_size = 1;
  int max_size = 12;
  bool little_endian = true;
  bool is_signed = false;
  double factor = 1.0;
  double offset = 0.0;
  double target_value = 0.0;
  FindSignalCompare compare = FindSignalCompare::Equal;
  std::string status;
  int history_index = -1;
  std::vector<FindSignalHistoryEntry> history;
};

constexpr size_t kFindSignalHistoryLimit = 6;
constexpr size_t kFindSignalMaxRows = 512;

FindSignalCompare find_signal_compare_from_token(std::string_view token);
FindSignalHistoryEntry parse_find_signal_history_entry(const json11::Json &item);
FindSignalPaneState parse_find_signal_pane_state(std::string_view state_json);
FindSignalHistoryEntry find_signal_history_entry_from_state(const FindSignalPaneState &state);

FindSignalHistoryEntry parse_find_signal_history_entry(const json11::Json &item) {
  FindSignalHistoryEntry entry;
  if (!item.is_object()) return entry;
  if (item["bus"].is_number()) entry.bus = std::clamp(item["bus"].int_value(), -1, 255);
  if (item["address"].is_string()) entry.address_hex = item["address"].string_value();
  if (item["min_size"].is_number()) entry.min_size = std::clamp(item["min_size"].int_value(), 1, 64);
  if (item["max_size"].is_number()) entry.max_size = std::clamp(item["max_size"].int_value(), 1, 64);
  if (item["little_endian"].is_bool()) entry.little_endian = item["little_endian"].bool_value();
  if (item["signed"].is_bool()) entry.is_signed = item["signed"].bool_value();
  if (item["factor"].is_number() && std::isfinite(item["factor"].number_value())) entry.factor = item["factor"].number_value();
  if (item["offset"].is_number() && std::isfinite(item["offset"].number_value())) entry.offset = item["offset"].number_value();
  if (item["target"].is_number() && std::isfinite(item["target"].number_value())) entry.target_value = item["target"].number_value();
  if (item["compare"].is_string()) entry.compare = find_signal_compare_from_token(item["compare"].string_value());
  if (entry.max_size < entry.min_size) entry.max_size = entry.min_size;
  return entry;
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

std::string trim_signal_name(std::string_view text) {
  size_t begin = 0;
  while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin]))) ++begin;
  size_t end = text.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) --end;
  return std::string(text.substr(begin, end - begin));
}

void push_history_entry(std::vector<FindSignalHistoryEntry> *history, const FindSignalHistoryEntry &entry) {
  if (history == nullptr) return;
  history->erase(std::remove_if(history->begin(), history->end(), [&](const FindSignalHistoryEntry &item) {
    return item.bus == entry.bus && item.address_hex == entry.address_hex && item.min_size == entry.min_size &&
           item.max_size == entry.max_size && item.little_endian == entry.little_endian &&
           item.is_signed == entry.is_signed && item.factor == entry.factor && item.offset == entry.offset &&
           item.target_value == entry.target_value && item.compare == entry.compare;
  }), history->end());
  history->push_back(entry);
  while (history->size() > kFindSignalHistoryLimit) history->erase(history->begin());
}

struct FindSignalPaneTransientState {
  FindSignalPaneState state;
  std::string state_json;
  bool valid = false;
  FindSignalJob job;
  std::vector<FindSignalResult> results;
};

FindSignalPaneTransientState &find_signal_transient_state(PaneInstance &pane) {
  if (FindSignalPaneTransientState *state = std::any_cast<FindSignalPaneTransientState>(&pane.transient_state)) return *state;
  pane.transient_state = FindSignalPaneTransientState{};
  return std::any_cast<FindSignalPaneTransientState &>(pane.transient_state);
}

FindSignalPaneState &find_signal_pane_state(PaneInstance &pane) {
  FindSignalPaneTransientState &transient = find_signal_transient_state(pane);
  if (!transient.valid || transient.state_json != pane.state_json) {
    transient.state = parse_find_signal_pane_state(pane.state_json);
    transient.state_json = pane.state_json;
    transient.valid = true;
    transient.job = FindSignalJob{};
    transient.results.clear();
  }
  return transient.state;
}

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

FindSignalHistoryEntry find_signal_history_entry_from_state(const FindSignalPaneState &state) {
  return {.bus = state.bus,
          .address_hex = state.address_hex,
          .min_size = state.min_size,
          .max_size = state.max_size,
          .little_endian = state.little_endian,
          .is_signed = state.is_signed,
          .factor = state.factor,
          .offset = state.offset,
          .target_value = state.target_value,
          .compare = state.compare};
}

void find_signal_apply_history_entry(FindSignalPaneState &state, const FindSignalHistoryEntry &entry) {
  state.bus = entry.bus;
  state.address_hex = entry.address_hex;
  state.min_size = entry.min_size;
  state.max_size = entry.max_size;
  state.little_endian = entry.little_endian;
  state.is_signed = entry.is_signed;
  state.factor = entry.factor;
  state.offset = entry.offset;
  state.target_value = entry.target_value;
  state.compare = entry.compare;
}

void find_signal_record_history_entry(FindSignalPaneState &state, size_t max_history = 6) {
  const FindSignalHistoryEntry entry = find_signal_history_entry_from_state(state);
  state.history.erase(std::remove_if(state.history.begin(), state.history.end(), [&](const FindSignalHistoryEntry &item) {
    return item.bus == entry.bus && item.address_hex == entry.address_hex && item.min_size == entry.min_size &&
           item.max_size == entry.max_size && item.little_endian == entry.little_endian &&
           item.is_signed == entry.is_signed && item.factor == entry.factor && item.offset == entry.offset &&
           item.target_value == entry.target_value && item.compare == entry.compare;
  }), state.history.end());
  state.history.push_back(entry);
  while (state.history.size() > max_history) state.history.erase(state.history.begin());
  state.history_index = static_cast<int>(state.history.size()) - 1;
}

std::string find_signal_history_entry_label(const FindSignalHistoryEntry &entry) {
  std::string label = (entry.bus >= 0 ? std::to_string(entry.bus) : std::string("any")) + ":";
  label += entry.address_hex.empty() ? std::string("*") : entry.address_hex;
  label += " ";
  label += std::to_string(entry.min_size) + "-" + std::to_string(entry.max_size);
  label += entry.little_endian ? " LE" : " BE";
  label += entry.is_signed ? " signed" : " unsigned";
  label += " ";
  label += find_signal_compare_label(entry.compare);
  label += " ";
  label += double_to_string(entry.target_value);
  return label;
}

FindSignalPaneState parse_find_signal_pane_state(std::string_view state_json) {
  FindSignalPaneState state;
  std::string err;
  const json11::Json json = json11::Json::parse(std::string(state_json), err);
  if (!err.empty() || !json.is_object()) return state;
  if (json["bus"].is_number()) state.bus = std::clamp(json["bus"].int_value(), -1, 255);
  if (json["address"].is_string()) state.address_hex = json["address"].string_value();
  if (json["signal_name"].is_string()) state.signal_name = json["signal_name"].string_value();
  if (json["min_size"].is_number()) state.min_size = std::clamp(json["min_size"].int_value(), 1, 64);
  if (json["max_size"].is_number()) state.max_size = std::clamp(json["max_size"].int_value(), 1, 64);
  if (json["little_endian"].is_bool()) state.little_endian = json["little_endian"].bool_value();
  if (json["signed"].is_bool()) state.is_signed = json["signed"].bool_value();
  if (json["factor"].is_number()) state.factor = json["factor"].number_value();
  if (json["offset"].is_number()) state.offset = json["offset"].number_value();
  if (json["target"].is_number()) state.target_value = json["target"].number_value();
  if (json["compare"].is_string()) state.compare = find_signal_compare_from_token(json["compare"].string_value());
  if (json["status"].is_string()) state.status = json["status"].string_value();
  if (json["history_index"].is_number()) state.history_index = json["history_index"].int_value();
  if (json["history"].is_array()) {
    for (const json11::Json &item : json["history"].array_items()) {
      if (!item.is_object()) continue;
      push_history_entry(&state.history, parse_find_signal_history_entry(item));
    }
  }
  state.history_index = std::clamp(state.history_index, -1, static_cast<int>(state.history.size()) - 1);
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
    {"signal_name", state.signal_name},
    {"min_size", state.min_size},
    {"max_size", state.max_size},
    {"little_endian", state.little_endian},
    {"signed", state.is_signed},
    {"factor", state.factor},
    {"offset", state.offset},
    {"target", state.target_value},
    {"compare", find_signal_compare_token(state.compare)},
    {"status", state.status},
    {"history_index", state.history_index},
    {"history", [&]() {
      json11::Json::array out;
      out.reserve(state.history.size());
      for (const FindSignalHistoryEntry &entry : state.history) {
        out.push_back(json11::Json::object{
          {"bus", entry.bus},
          {"address", entry.address_hex},
          {"min_size", entry.min_size},
          {"max_size", entry.max_size},
          {"little_endian", entry.little_endian},
          {"signed", entry.is_signed},
          {"factor", entry.factor},
          {"offset", entry.offset},
          {"target", entry.target_value},
          {"compare", find_signal_compare_token(entry.compare)},
        });
      }
      return json11::Json(out);
    }()},
  }).dump();
}

FindSignalParams find_signal_params_from_state(const FindSignalPaneState &state, TimeRange range) {
  FindSignalParams params;
  params.max_results = kFindSignalMaxRows;
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

std::string find_signal_default_name(const FindSignalResult &result) {
  char buf[96];
  std::snprintf(buf, sizeof(buf), "SIG_%X_%02d_%02d", result.id.address, result.sig.start_bit, result.sig.size);
  return buf;
}

}  // namespace

[[maybe_unused]] bool commit_find_signal_result(Session &session, std::string_view selection_group,
                               const FindSignalResult &result, std::string_view signal_name,
                               std::string &error) {
  if (result.msg_size == 0) {
    error = "candidate has no message size";
    return false;
  }
  Signal signal = result.sig;
  signal.name = trim_signal_name(signal_name);
  if (signal.name.empty()) signal.name = find_signal_default_name(result);
  std::string local_error;
  if (!commit_signal_add(session.dbc_undo, session.dbc, result.id, std::move(signal), result.msg_size, local_error)) {
    error = std::move(local_error);
    return false;
  }
  error.clear();
  SelectionContext &selection = session.selection(selection_group);
  selection.selected_msg_id = result.id;
  selection.has_selected_msg = true;
  return true;
}

void draw_find_signal_pane(Session &session, PaneInstance &pane) {
  FindSignalPaneTransientState &transient = find_signal_transient_state(pane);
  FindSignalJob &job = transient.job;
  std::vector<FindSignalResult> &results = transient.results;
  FindSignalPaneState &state = find_signal_pane_state(pane);
  bool changed = false;
  state.history_index = std::clamp(state.history_index, -1, static_cast<int>(state.history.size()) - 1);

  ImGui::SetNextItemWidth(70.0f);
  changed |= ImGui::InputInt("Bus", &state.bus);
  if (ImGui::GetContentRegionAvail().x > 120.0f) ImGui::SameLine();
  ImGui::SetNextItemWidth(90.0f);
  changed |= input_text_with_hint("Addr", "", &state.address_hex);
  if (ImGui::GetContentRegionAvail().x > 144.0f) ImGui::SameLine();
  ImGui::SetNextItemWidth(128.0f);
  changed |= input_text_with_hint("Name", "", &state.signal_name);

  // Min/Max always start a fresh row: Bus/Addr/Name alone can already fill a narrow pane,
  // and gambling their combined width against one more SameLine leaves Max unreachable.
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

  if (!state.history.empty()) {
    const std::vector<std::string> history_labels = [&]() {
      std::vector<std::string> labels;
      labels.reserve(state.history.size());
      for (const FindSignalHistoryEntry &entry : state.history) labels.push_back(find_signal_history_entry_label(entry));
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
      find_signal_apply_history_entry(state, state.history[static_cast<size_t>(std::clamp(state.history_index, 0, static_cast<int>(state.history.size()) - 1))]);
      changed = true;
    }
  }

  if (ImGui::Button(job.done ? "Search" : "Restart")) {
    find_signal_record_history_entry(state);
    job = make_find_signal_job(session.store, find_signal_params_from_state(state, session.view_range.range()));
    results.clear();
    state.status = "Scanning";
    changed = true;
  }
  if (!job.done && job.store == &session.store) {
    step_find_signal_job(job, 512);
    results = job.results;
    state.status = job.done ? ("Found " + std::to_string(results.size()) + " candidates") :
                              ("Scanning " + std::to_string(results.size()) + " candidates");
    changed = true;
  }
  ImGui::SameLine();
  ImGui::TextDisabled("%s", state.status.c_str());

  if (changed) {
    pane.state_json = find_signal_pane_state_json(state);
    transient.state_json = pane.state_json;
  }

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
        ImGui::TextUnformatted(result.id.to_string().c_str());
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
          state.status = commit_find_signal_result(session, pane.selection_group, result, state.signal_name, error)
                           ? "Created DBC signal"
                           : error;
          pane.state_json = find_signal_pane_state_json(state);
          transient.state_json = pane.state_json;
        }
        ImGui::PopID();
      }
    }
    ImGui::EndTable();
  }
}

}  // namespace loggy
