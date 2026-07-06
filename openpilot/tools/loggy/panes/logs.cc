#include "tools/loggy/panes/logs.h"

#include "tools/loggy/backend/session.h"
#include "tools/loggy/backend/route.h"
#include "tools/loggy/shell/theme.h"
#include "tools/loggy/shell/workspace.h"

#include "imgui.h"
#include "json11/json11.hpp"

#include <algorithm>
#include <any>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <string>

namespace loggy {
namespace {

inline constexpr uint32_t kLogLevelTraceMask = 1u << 0;
inline constexpr uint32_t kLogLevelInfoMask = 1u << 1;
inline constexpr uint32_t kLogLevelWarnMask = 1u << 2;
inline constexpr uint32_t kLogLevelErrorMask = 1u << 3;
inline constexpr uint32_t kLogLevelCriticalMask = 1u << 4;
inline constexpr uint32_t kLogLevelAllMask = kLogLevelTraceMask | kLogLevelInfoMask |
                                             kLogLevelWarnMask | kLogLevelErrorMask |
                                             kLogLevelCriticalMask;

struct LogPaneState {
  std::string filter;
  std::string source_filter;
  uint8_t min_level = 0;
  uint32_t level_mask = kLogLevelAllMask;
  int origin_filter = -1;
  int time_mode = 0;
  bool follow = true;
  size_t max_rows = 500;
  int selected_log_index = -1;
};

struct LogFilterParams {
  std::string filter;
  std::string source_filter;
  uint8_t min_level = 0;
  uint32_t level_mask = kLogLevelAllMask;
  int origin_filter = -1;
  size_t max_rows = 500;
};

LogPaneState parse_logs_pane_state(std::string_view state_json);
std::string logs_pane_state_json(const LogPaneState &state);
const char *log_origin_label(LogOrigin origin);
const char *log_level_label(uint8_t level);
uint32_t log_level_mask_bit(uint8_t level);
uint32_t log_level_mask_from_min_level(uint8_t min_level);
std::string log_level_mask_label(uint32_t mask);
const char *log_time_mode_label(int mode);
std::string log_time_text(const LogEntry &entry, int mode);
std::string log_detail_text(const LogEntry &entry);
int log_selected_row_position(const std::vector<size_t> &rows, int selected_log_index);
int log_navigate_selected_row(const std::vector<size_t> &rows, int selected_log_index, int direction);
std::vector<size_t> filter_log_entries(const std::vector<LogEntry> &logs,
                                       const LogFilterParams &params);

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

struct LogPaneRuntimeState {
  LogPaneState state;
  std::string loaded_json;
};

LogPaneState &logs_pane_state(PaneInstance *pane) {
  auto *runtime = std::any_cast<LogPaneRuntimeState>(&pane->transient_state);
  if (runtime == nullptr || runtime->loaded_json != pane->state_json) {
    pane->transient_state = LogPaneRuntimeState{
      .state = parse_logs_pane_state(pane->state_json),
      .loaded_json = pane->state_json,
    };
    runtime = std::any_cast<LogPaneRuntimeState>(&pane->transient_state);
  }
  return runtime->state;
}

void save_logs_pane_state(PaneInstance *pane, const LogPaneState &state) {
  pane->state_json = logs_pane_state_json(state);
  auto *runtime = std::any_cast<LogPaneRuntimeState>(&pane->transient_state);
  if (runtime != nullptr) {
    runtime->state = state;
    runtime->loaded_json = pane->state_json;
  }
}

bool log_entry_matches(const LogEntry &entry, const LogFilterParams &params) {
  if (params.level_mask == kLogLevelAllMask) {
    if (entry.level < params.min_level) return false;
  } else if ((log_level_mask_bit(entry.level) & params.level_mask) == 0) {
    return false;
  }
  if (params.origin_filter >= 0 && static_cast<int>(entry.origin) != params.origin_filter) return false;
  if (!params.source_filter.empty() && !contains_case_insensitive(entry.source, params.source_filter)) return false;
  if (params.filter.empty()) return true;
  return contains_case_insensitive(entry.message, params.filter)
      || contains_case_insensitive(entry.source, params.filter)
      || contains_case_insensitive(entry.func, params.filter)
      || contains_case_insensitive(entry.context, params.filter);
}

bool draw_level_mask_popup(LogPaneState *state) {
  if (state == nullptr) return false;
  struct LevelOption {
    const char *label;
    uint32_t mask;
  };
  constexpr std::array<LevelOption, 5> levels = {{
    {"Trace", kLogLevelTraceMask},
    {"Info", kLogLevelInfoMask},
    {"Warn", kLogLevelWarnMask},
    {"Error", kLogLevelErrorMask},
    {"Critical", kLogLevelCriticalMask},
  }};

  bool changed = false;
  const std::string label = log_level_mask_label(state->level_mask);
  if (ImGui::Button(label.c_str(), ImVec2(118.0f, 0.0f))) ImGui::OpenPopup("##log_level_mask");
  if (ImGui::BeginPopup("##log_level_mask")) {
    if (ImGui::Button("All")) {
      state->level_mask = kLogLevelAllMask;
      state->min_level = 0;
      changed = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("None")) {
      state->level_mask = 0;
      state->min_level = 0;
      changed = true;
    }

    for (const LevelOption &option : levels) {
      bool enabled = (state->level_mask & option.mask) != 0;
      if (ImGui::Checkbox(option.label, &enabled)) {
        if (enabled) {
          state->level_mask |= option.mask;
        } else {
          state->level_mask &= ~option.mask;
        }
        state->min_level = 0;
        changed = true;
      }
    }
    ImGui::EndPopup();
  }
  return changed;
}

void draw_origin_combo(LogPaneState *state) {
  struct OriginOption {
    const char *label;
    int origin;
  };
  constexpr std::array<OriginOption, 4> origins = {{
    {"Any", -1},
    {"Log", static_cast<int>(LogOrigin::Log)},
    {"OS", static_cast<int>(LogOrigin::OperatingSystem)},
    {"Alert", static_cast<int>(LogOrigin::Alert)},
  }};

  const char *label = "Any";
  for (const OriginOption &option : origins) {
    if (option.origin == state->origin_filter) {
      label = option.label;
      break;
    }
  }

  ImGui::SetNextItemWidth(86.0f);
  if (ImGui::BeginCombo("##log_origin", label)) {
    for (const OriginOption &option : origins) {
      const bool selected = option.origin == state->origin_filter;
      if (ImGui::Selectable(option.label, selected)) state->origin_filter = option.origin;
      if (selected) ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }
}

void draw_time_mode_combo(LogPaneState *state) {
  ImGui::SetNextItemWidth(86.0f);
  if (ImGui::BeginCombo("##log_time", log_time_mode_label(state->time_mode))) {
    for (int mode = 0; mode < 3; ++mode) {
      const bool selected = mode == state->time_mode;
      if (ImGui::Selectable(log_time_mode_label(mode), selected)) state->time_mode = mode;
      if (selected) ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }
}

LogPaneState parse_logs_pane_state(std::string_view state_json) {
  LogPaneState state;
  if (state_json.empty()) return state;
  std::string err;
  const json11::Json json = json11::Json::parse(std::string(state_json), err);
  if (!err.empty() || !json.is_object()) return state;
  if (json["filter"].is_string()) state.filter = json["filter"].string_value();
  if (json["source_filter"].is_string()) state.source_filter = json["source_filter"].string_value();
  if (json["min_level"].is_number()) {
    state.min_level = static_cast<uint8_t>(std::clamp(json["min_level"].int_value(), 0, 255));
    state.level_mask = log_level_mask_from_min_level(state.min_level);
  }
  if (json["level_mask"].is_number()) {
    const int raw_mask = json["level_mask"].int_value();
    state.level_mask = static_cast<uint32_t>(std::max(raw_mask, 0)) & kLogLevelAllMask;
    state.min_level = 0;
  }
  if (json["origin"].is_number()) state.origin_filter = std::clamp(json["origin"].int_value(), -1, 2);
  if (json["time_mode"].is_number()) state.time_mode = std::clamp(json["time_mode"].int_value(), 0, 2);
  if (json["follow"].is_bool()) state.follow = json["follow"].bool_value();
  if (json["max_rows"].is_number()) {
    state.max_rows = static_cast<size_t>(std::clamp(json["max_rows"].int_value(), 1, 5000));
  }
  if (json["selected_log_index"].is_number()) {
    state.selected_log_index = std::max(json["selected_log_index"].int_value(), -1);
  }
  return state;
}

std::string logs_pane_state_json(const LogPaneState &state) {
  return json11::Json(json11::Json::object{
    {"filter", state.filter},
    {"source_filter", state.source_filter},
    {"min_level", static_cast<int>(state.min_level)},
    {"level_mask", static_cast<int>(state.level_mask & kLogLevelAllMask)},
    {"origin", state.origin_filter},
    {"time_mode", state.time_mode},
    {"follow", state.follow},
    {"max_rows", static_cast<int>(state.max_rows)},
    {"selected_log_index", state.selected_log_index},
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

uint32_t log_level_mask_bit(uint8_t level) {
  if (level >= 50) return kLogLevelCriticalMask;
  if (level >= 40) return kLogLevelErrorMask;
  if (level >= 30) return kLogLevelWarnMask;
  if (level >= 20) return kLogLevelInfoMask;
  return kLogLevelTraceMask;
}

uint32_t log_level_mask_from_min_level(uint8_t min_level) {
  if (min_level >= 50) return kLogLevelCriticalMask;
  if (min_level >= 40) return kLogLevelErrorMask | kLogLevelCriticalMask;
  if (min_level >= 30) return kLogLevelWarnMask | kLogLevelErrorMask | kLogLevelCriticalMask;
  if (min_level >= 20) return kLogLevelInfoMask | kLogLevelWarnMask | kLogLevelErrorMask | kLogLevelCriticalMask;
  return kLogLevelAllMask;
}

std::string log_level_mask_label(uint32_t mask) {
  mask &= kLogLevelAllMask;
  if (mask == kLogLevelAllMask) return "All";
  if (mask == 0) return "None";
  std::string out;
  auto append = [&](uint32_t bit, const char *label) {
    if ((mask & bit) == 0) return;
    if (!out.empty()) out += ",";
    out += label;
  };
  append(kLogLevelTraceMask, "Trace");
  append(kLogLevelInfoMask, "Info");
  append(kLogLevelWarnMask, "Warn");
  append(kLogLevelErrorMask, "Error");
  append(kLogLevelCriticalMask, "Crit");
  return out;
}

const char *log_time_mode_label(int mode) {
  switch (mode) {
    case 1: return "Boot";
    case 2: return "Wall";
    case 0:
    default: return "Route";
  }
}

std::string log_time_text(const LogEntry &entry, int mode) {
  char buf[64];
  const double time = mode == 1 ? entry.boot_time : mode == 2 ? entry.wall_time : entry.mono_time;
  if (mode == 2 && time > 0.0) {
    std::snprintf(buf, sizeof(buf), "%.0f", time);
  } else {
    std::snprintf(buf, sizeof(buf), "%.2f", time);
  }
  return buf;
}

std::string log_detail_text(const LogEntry &entry) {
  std::string out;
  if (!entry.func.empty()) out += "func: " + entry.func;
  if (!entry.context.empty()) {
    if (!out.empty()) out += "\n";
    out += "context: " + entry.context;
  }
  return out;
}

int log_selected_row_position(const std::vector<size_t> &rows, int selected_log_index) {
  if (selected_log_index < 0) return 0;
  const size_t selected = static_cast<size_t>(selected_log_index);
  for (size_t i = 0; i < rows.size(); ++i) {
    if (rows[i] == selected) return static_cast<int>(i + 1);
  }
  return 0;
}

int log_navigate_selected_row(const std::vector<size_t> &rows, int selected_log_index, int direction) {
  if (rows.empty()) return -1;
  const int current_position = log_selected_row_position(rows, selected_log_index);
  if (current_position == 0) {
    return direction < 0 ? static_cast<int>(rows.back()) : static_cast<int>(rows.front());
  }
  const int row_count = static_cast<int>(rows.size());
  const int current_index = current_position - 1;
  const int next_index = direction < 0
      ? (current_index + row_count - 1) % row_count
      : (current_index + 1) % row_count;
  return static_cast<int>(rows[static_cast<size_t>(next_index)]);
}

std::vector<size_t> filter_log_entries(const std::vector<LogEntry> &logs,
                                       const LogFilterParams &params) {
  std::vector<size_t> rows;
  if (params.max_rows == 0) return rows;
  rows.reserve(std::min(params.max_rows, logs.size()));
  for (size_t i = 0; i < logs.size(); ++i) {
    if (!log_entry_matches(logs[i], params)) continue;
    rows.push_back(i);
    if (rows.size() >= params.max_rows) break;
  }
  return rows;
}

}  // namespace

void draw_logs_pane(Session &session, PaneInstance &pane) {
  LogPaneState &state = logs_pane_state(&pane);
  const uint8_t old_min_level = state.min_level;
  const uint32_t old_level_mask = state.level_mask;
  const int old_origin_filter = state.origin_filter;
  const int old_time_mode = state.time_mode;
  const int old_selected_log_index = state.selected_log_index;

  ImGui::SetNextItemWidth(std::min(260.0f, std::max(120.0f, ImGui::GetContentRegionAvail().x * 0.40f)));
  bool changed = input_text_with_hint("##log_filter", "Filter logs", &state.filter);
  ImGui::SameLine();
  ImGui::SetNextItemWidth(std::min(180.0f, std::max(120.0f, ImGui::GetContentRegionAvail().x * 0.38f)));
  changed = input_text_with_hint("##log_source", "Source", &state.source_filter) || changed;

  ImGui::TextDisabled("Level");
  ImGui::SameLine();
  changed = draw_level_mask_popup(&state) || changed;
  ImGui::SameLine();
  ImGui::TextDisabled("Origin");
  ImGui::SameLine();
  draw_origin_combo(&state);
  ImGui::SameLine();
  ImGui::TextDisabled("Time");
  ImGui::SameLine();
  draw_time_mode_combo(&state);
  ImGui::SameLine();
  changed = ImGui::Checkbox("Follow", &state.follow) || changed;
  const std::vector<LogEntry> &logs = session.logs;
  const std::vector<size_t> rows = filter_log_entries(logs, LogFilterParams{
    .filter = state.filter,
    .source_filter = state.source_filter,
    .min_level = state.min_level,
    .level_mask = state.level_mask,
    .origin_filter = state.origin_filter,
    .max_rows = state.max_rows,
  });
  ImGui::SameLine();
  ImGui::TextDisabled("%zu/%zu logs", rows.size(), logs.size());

  int scroll_to_log_index = -1;
  ImGui::SameLine();
  ImGui::BeginDisabled(rows.empty());
  if (ImGui::SmallButton("Prev")) {
    state.selected_log_index = log_navigate_selected_row(rows, state.selected_log_index, -1);
    state.follow = false;
    scroll_to_log_index = state.selected_log_index;
    changed = true;
  }
  ImGui::SameLine();
  if (ImGui::SmallButton("Next")) {
    state.selected_log_index = log_navigate_selected_row(rows, state.selected_log_index, 1);
    state.follow = false;
    scroll_to_log_index = state.selected_log_index;
    changed = true;
  }
  ImGui::EndDisabled();
  const int selected_position = log_selected_row_position(rows, state.selected_log_index);
  ImGui::SameLine();
  ImGui::TextDisabled("%d/%zu", selected_position, rows.size());
  if (state.selected_log_index >= 0) {
    ImGui::SameLine();
    if (ImGui::SmallButton("Clear")) {
      state.selected_log_index = -1;
      changed = true;
    }
  }

  changed = changed || old_min_level != state.min_level ||
            old_level_mask != state.level_mask ||
            old_origin_filter != state.origin_filter ||
            old_time_mode != state.time_mode ||
            old_selected_log_index != state.selected_log_index;
  if (changed) {
    save_logs_pane_state(&pane, state);
  }

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
    const bool selected = state.selected_log_index >= 0 &&
                          row_index == static_cast<size_t>(state.selected_log_index);
    if (selected) {
      ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, ImGui::GetColorU32(ImGuiCol_HeaderHovered));
    }
    ImGui::TableSetColumnIndex(0);
    ImGui::TextUnformatted(log_time_text(entry, state.time_mode).c_str());
    ImGui::TableSetColumnIndex(1);
    ImGui::TextUnformatted(log_level_label(entry.level));
    ImGui::TableSetColumnIndex(2);
    ImGui::TextUnformatted(log_origin_label(entry.origin));
    ImGui::TableSetColumnIndex(3);
    ImGui::TextUnformatted(entry.source.empty() ? "-" : entry.source.c_str());
    ImGui::TableSetColumnIndex(4);
    const char *message = entry.message.empty() ? entry.context.c_str() : entry.message.c_str();
    const std::string detail = log_detail_text(entry);
    if (detail.empty()) {
      ImGui::TextUnformatted(message);
    } else {
      ImGui::PushID(static_cast<int>(row_index));
      const bool open = ImGui::TreeNodeEx("##log_detail", ImGuiTreeNodeFlags_SpanAvailWidth, "%s", message);
      if (open) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(4);
        ImGui::PushTextWrapPos();
        ImGui::TextUnformatted(detail.c_str());
        ImGui::PopTextWrapPos();
        ImGui::TreePop();
      }
      ImGui::PopID();
    }
    if (scroll_to_log_index >= 0 && row_index == static_cast<size_t>(scroll_to_log_index)) {
      ImGui::SetScrollHereY(0.5f);
    }
  }
  pop_mono_font();

  if (state.follow && !rows.empty()) ImGui::SetScrollHereY(1.0f);
  ImGui::EndTable();
}

}  // namespace loggy
