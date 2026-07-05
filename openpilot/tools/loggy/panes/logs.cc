#include "tools/loggy/panes/logs.h"

#include "tools/loggy/backend/session.h"
#include "tools/loggy/shell/theme.h"
#include "tools/loggy/shell/workspace.h"

#include "imgui.h"
#include "json11/json11.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <string>

namespace loggy {
namespace {

std::string lower_copy(std::string_view value) {
  std::string out(value);
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}

bool contains_case_insensitive(std::string_view haystack, std::string_view needle) {
  if (needle.empty()) return true;
  return lower_copy(haystack).find(lower_copy(needle)) != std::string::npos;
}

bool log_entry_matches(const LogEntry &entry, std::string_view filter, uint8_t min_level) {
  if (entry.level < min_level) return false;
  if (filter.empty()) return true;
  return contains_case_insensitive(entry.message, filter)
      || contains_case_insensitive(entry.source, filter)
      || contains_case_insensitive(entry.func, filter)
      || contains_case_insensitive(entry.context, filter);
}

void draw_level_combo(LogPaneState *state) {
  struct LevelOption {
    const char *label;
    uint8_t level;
  };
  constexpr std::array<LevelOption, 5> levels = {{
    {"All", 0},
    {"Info+", 20},
    {"Warn+", 30},
    {"Error+", 40},
    {"Critical", 50},
  }};

  const char *label = "All";
  for (const LevelOption &option : levels) {
    if (option.level == state->min_level) {
      label = option.label;
      break;
    }
  }

  ImGui::SetNextItemWidth(92.0f);
  if (ImGui::BeginCombo("##log_level", label)) {
    for (const LevelOption &option : levels) {
      const bool selected = option.level == state->min_level;
      if (ImGui::Selectable(option.label, selected)) state->min_level = option.level;
      if (selected) ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }
}

}  // namespace

LogPaneState parse_logs_pane_state(std::string_view state_json) {
  LogPaneState state;
  if (state_json.empty()) return state;
  std::string err;
  const json11::Json json = json11::Json::parse(std::string(state_json), err);
  if (!err.empty() || !json.is_object()) return state;
  if (json["filter"].is_string()) state.filter = json["filter"].string_value();
  if (json["min_level"].is_number()) {
    state.min_level = static_cast<uint8_t>(std::clamp(json["min_level"].int_value(), 0, 255));
  }
  if (json["follow"].is_bool()) state.follow = json["follow"].bool_value();
  if (json["max_rows"].is_number()) {
    state.max_rows = static_cast<size_t>(std::clamp(json["max_rows"].int_value(), 1, 5000));
  }
  return state;
}

std::string logs_pane_state_json(const LogPaneState &state) {
  return json11::Json(json11::Json::object{
    {"filter", state.filter},
    {"min_level", static_cast<int>(state.min_level)},
    {"follow", state.follow},
    {"max_rows", static_cast<int>(state.max_rows)},
  }).dump();
}

const char *log_origin_label(LogOrigin origin) {
  switch (origin) {
    case LogOrigin::OperatingSystem: return "OS";
    case LogOrigin::Alert: return "Alert";
    case LogOrigin::Log:
    default:
      return "Log";
  }
}

const char *log_level_label(uint8_t level) {
  if (level >= 50) return "CRIT";
  if (level >= 40) return "ERROR";
  if (level >= 30) return "WARN";
  if (level >= 20) return "INFO";
  return "TRACE";
}

std::vector<size_t> filter_log_entries(const std::vector<LogEntry> &logs,
                                       std::string_view filter,
                                       uint8_t min_level,
                                       size_t max_rows) {
  std::vector<size_t> rows;
  if (max_rows == 0) return rows;
  rows.reserve(std::min(max_rows, logs.size()));
  for (size_t i = 0; i < logs.size(); ++i) {
    if (!log_entry_matches(logs[i], filter, min_level)) continue;
    rows.push_back(i);
    if (rows.size() >= max_rows) break;
  }
  return rows;
}

void draw_logs_pane(Session &session, PaneInstance &pane) {
  LogPaneState state = parse_logs_pane_state(pane.state_json);
  char filter_buf[160] = {};
  std::snprintf(filter_buf, sizeof(filter_buf), "%s", state.filter.c_str());

  ImGui::SetNextItemWidth(std::min(260.0f, std::max(120.0f, ImGui::GetContentRegionAvail().x * 0.35f)));
  bool changed = ImGui::InputTextWithHint("##log_filter", "Filter logs", filter_buf, sizeof(filter_buf));
  ImGui::SameLine();
  draw_level_combo(&state);
  ImGui::SameLine();
  changed = ImGui::Checkbox("Follow", &state.follow) || changed;
  if (changed) {
    state.filter = filter_buf;
    pane.state_json = logs_pane_state_json(state);
  }

  const std::vector<LogEntry> &logs = session.logs();
  const std::vector<size_t> rows = filter_log_entries(logs, state.filter, state.min_level, state.max_rows);
  ImGui::SameLine();
  ImGui::TextDisabled("%zu/%zu logs", rows.size(), logs.size());

  constexpr ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
                                    ImGuiTableFlags_BordersOuter | ImGuiTableFlags_SizingStretchProp |
                                    ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable;
  if (!ImGui::BeginTable("##loggy_logs", 5, flags, ImGui::GetContentRegionAvail())) return;
  ImGui::TableSetupScrollFreeze(0, 1);
  ImGui::TableSetupColumn("Time", ImGuiTableColumnFlags_WidthFixed, 72.0f);
  ImGui::TableSetupColumn("Level", ImGuiTableColumnFlags_WidthFixed, 64.0f);
  ImGui::TableSetupColumn("Origin", ImGuiTableColumnFlags_WidthFixed, 70.0f);
  ImGui::TableSetupColumn("Source", ImGuiTableColumnFlags_WidthStretch, 0.75f);
  ImGui::TableSetupColumn("Message", ImGuiTableColumnFlags_WidthStretch, 2.25f);
  ImGui::TableHeadersRow();

  push_mono_font();
  for (size_t row_index : rows) {
    const LogEntry &entry = logs[row_index];
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("%.2f", entry.mono_time);
    ImGui::TableSetColumnIndex(1);
    ImGui::TextUnformatted(log_level_label(entry.level));
    ImGui::TableSetColumnIndex(2);
    ImGui::TextUnformatted(log_origin_label(entry.origin));
    ImGui::TableSetColumnIndex(3);
    ImGui::TextUnformatted(entry.source.empty() ? "-" : entry.source.c_str());
    ImGui::TableSetColumnIndex(4);
    ImGui::TextUnformatted(entry.message.empty() ? entry.context.c_str() : entry.message.c_str());
  }
  pop_mono_font();

  if (state.follow && !rows.empty()) ImGui::SetScrollHereY(1.0f);
  ImGui::EndTable();
}

}  // namespace loggy
