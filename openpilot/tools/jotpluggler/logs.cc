#include "tools/jotpluggler/app.h"

#include <cmath>
#include <ctime>

namespace {

struct LevelOption {
  const char *label;
  int value;
};

constexpr std::array<LevelOption, 5> LEVEL_OPTIONS = {{
  {"DEBUG", 10},
  {"INFO", 20},
  {"WARNING", 30},
  {"ERROR", 40},
  {"CRITICAL", 50},
}};
constexpr uint32_t ALL_LEVEL_MASK = (1u << LEVEL_OPTIONS.size()) - 1u;

bool log_matches_search(const LogEntry &entry, std::string_view query) {
  if (query.empty()) return true;
  const std::string needle = lowercase_copy(query);
  const auto contains = [&](std::string_view haystack) {
    return lowercase_copy(haystack).find(needle) != std::string::npos;
  };
  return contains(entry.message) || contains(entry.source) || contains(entry.func);
}

std::vector<std::string> collect_log_sources(const std::vector<LogEntry> &logs) {
  std::vector<std::string> sources;
  for (const LogEntry &entry : logs) {
    if (entry.source.empty()) continue;
    if (std::find(sources.begin(), sources.end(), entry.source) == sources.end()) {
      sources.push_back(entry.source);
    }
  }
  std::sort(sources.begin(), sources.end());
  return sources;
}

std::vector<int> filter_log_indices(const RouteData &route_data, const LogsUiState &logs_state) {
  std::vector<int> indices;
  indices.reserve(route_data.logs.size());
  for (size_t i = 0; i < route_data.logs.size(); ++i) {
    const LogEntry &entry = route_data.logs[i];
    int level_index = 0;
    if (entry.level >= 50) {
      level_index = 4;
    } else if (entry.level >= 40) {
      level_index = 3;
    } else if (entry.level >= 30) {
      level_index = 2;
    } else if (entry.level >= 20) {
      level_index = 1;
    }
    if ((logs_state.enabled_levels_mask & (1u << level_index)) == 0) {
      continue;
    }
    if (!logs_state.all_sources) {
      const auto it = std::find(logs_state.selected_sources.begin(),
                                logs_state.selected_sources.end(),
                                entry.source);
      if (it == logs_state.selected_sources.end()) continue;
    }
    if (!log_matches_search(entry, logs_state.search)) continue;
    indices.push_back(static_cast<int>(i));
  }
  return indices;
}

int find_active_log_position(const RouteData &route_data,
                             const std::vector<int> &filtered_indices,
                             double tracker_time) {
  if (filtered_indices.empty()) return -1;
  auto it = std::lower_bound(filtered_indices.begin(), filtered_indices.end(), tracker_time,
                             [&](int log_index, double tm) {
                               return route_data.logs[static_cast<size_t>(log_index)].mono_time < tm;
                             });
  if (it == filtered_indices.begin()) return static_cast<int>(std::distance(filtered_indices.begin(), it));
  if (it == filtered_indices.end()) return static_cast<int>(filtered_indices.size()) - 1;
  if (route_data.logs[static_cast<size_t>(*it)].mono_time > tracker_time) {
    --it;
  }
  return static_cast<int>(std::distance(filtered_indices.begin(), it));
}

std::string format_route_time(double seconds) {
  if (seconds < 0.0) {
    seconds = 0.0;
  }
  const int minutes = static_cast<int>(seconds / 60.0);
  const double remaining = seconds - static_cast<double>(minutes) * 60.0;
  return util::string_format("%d:%06.3f", minutes, remaining);
}

std::string format_boot_time(double seconds) {
  return util::string_format("%.3f", seconds);
}

std::string format_wall_time(double seconds) {
  if (seconds <= 0.0) return "--";
  const time_t wall_seconds = static_cast<time_t>(seconds);
  std::tm wall_tm = {};
  localtime_r(&wall_seconds, &wall_tm);
  const int millis = static_cast<int>(std::llround((seconds - std::floor(seconds)) * 1000.0));
  return util::string_format("%02d:%02d:%02d.%03d",
                             wall_tm.tm_hour, wall_tm.tm_min, wall_tm.tm_sec, millis);
}

std::string format_log_time(const LogEntry &entry, LogTimeMode mode) {
  switch (mode) {
    case LogTimeMode::Route:
      return format_route_time(entry.mono_time);
    case LogTimeMode::Boot:
      return format_boot_time(entry.boot_time);
    case LogTimeMode::WallClock:
      return format_wall_time(entry.wall_time);
  }
  return format_route_time(entry.mono_time);
}

const char *time_mode_label(LogTimeMode mode) {
  switch (mode) {
    case LogTimeMode::Route: return "Route";
    case LogTimeMode::Boot: return "Boot";
    case LogTimeMode::WallClock: return "Wall clock";
  }
  return "Route";
}

std::string level_filter_label(uint32_t mask) {
  if (mask == ALL_LEVEL_MASK) return "All levels";
  if (mask == 0b11110) return "INFO+";
  if (mask == 0b11100) return "WARNING+";
  if (mask == 0b11000) return "ERROR+";
  if (mask == 0b10000) return "CRITICAL";

  int enabled_count = 0;
  const char *last_label = "None";
  for (size_t i = 0; i < LEVEL_OPTIONS.size(); ++i) {
    if ((mask & (1u << i)) == 0) {
      continue;
    }
    ++enabled_count;
    last_label = LEVEL_OPTIONS[i].label;
  }
  if (enabled_count == 0) return "None";
  if (enabled_count == 1) return last_label;
  return "Custom";
}

std::string source_filter_label(const LogsUiState &logs_state, const std::vector<std::string> &sources) {
  if (logs_state.all_sources || logs_state.selected_sources.size() == sources.size()) {
    return "All sources";
  }
  if (logs_state.selected_sources.empty()) return "No sources";
  if (logs_state.selected_sources.size() == 1) return logs_state.selected_sources.front();
  return std::to_string(logs_state.selected_sources.size()) + " sources";
}

const char *level_label(const LogEntry &entry) {
  if (entry.origin == LogOrigin::Alert) return "ALRT";
  if (entry.level >= 50) return "CRIT";
  if (entry.level >= 40) return "ERR";
  if (entry.level >= 30) return "WARN";
  if (entry.level >= 20) return "INFO";
  return "DBG";
}

ImVec4 level_text_color(const LogEntry &entry, bool active) {
  if (active) return color_rgb(46, 54, 63);
  if (entry.origin == LogOrigin::Alert) return color_rgb(50, 100, 200);
  if (entry.level >= 50) return color_rgb(176, 26, 18);
  if (entry.level >= 40) return color_rgb(200, 50, 40);
  if (entry.level >= 30) return color_rgb(200, 130, 0);
  if (entry.level >= 20) return color_rgb(80, 86, 94);
  return color_rgb(126, 133, 141);
}

ImU32 row_bg_color(const LogEntry &entry, bool active) {
  if (active) return IM_COL32(80, 140, 210, 38);
  return 0;
}

void set_tracker_to_log(UiState *state, const LogEntry &entry) {
  state->tracker_time = entry.mono_time;
  state->has_tracker_time = true;
  state->logs.last_auto_scroll_time = entry.mono_time;
}

void draw_log_expansion_row(const LogEntry &entry) {
  ImGui::TableNextRow();
  ImGui::TableSetColumnIndex(0);
  ImGui::TextUnformatted("");
  ImGui::TableSetColumnIndex(1);
  ImGui::TextUnformatted("");
  ImGui::TableSetColumnIndex(2);
  ImGui::TextUnformatted(entry.func.empty() ? "" : entry.func.c_str());
  ImGui::TableSetColumnIndex(3);
  ImGui::PushStyleColor(ImGuiCol_Text, color_rgb(96, 104, 113));
  ImGui::TextWrapped("%s", entry.message.c_str());
  if (!entry.func.empty()) {
    ImGui::TextWrapped("func: %s", entry.func.c_str());
  }
  if (!entry.context.empty()) {
    ImGui::TextWrapped("ctx: %s", entry.context.c_str());
  }
  ImGui::PopStyleColor();
}

void draw_log_row(const LogEntry &entry,
                  int log_index,
                  bool active,
                  UiState *state) {
  ImGui::PushID(log_index);
  const ImU32 bg = row_bg_color(entry, active);
  ImGui::TableNextRow();
  if (bg != 0) {
    ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, bg);
  }

  const std::string time_text = std::string(active ? "\xE2\x96\xB6 " : "  ") + format_log_time(entry, state->logs.time_mode);
  const auto clickable_text = [&](const char *id, const std::string &text, ImVec4 color = color_rgb(74, 80, 88)) {
    ImGui::PushID(id);
    ImGui::PushStyleColor(ImGuiCol_Text, color);
    ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(0, 0, 0, 0));
    const bool clicked = ImGui::Selectable(text.c_str(), false, ImGuiSelectableFlags_AllowDoubleClick);
    ImGui::PopStyleColor(4);
    ImGui::PopID();
    return clicked;
  };

  bool clicked = false;
  ImGui::TableSetColumnIndex(0);
  app_push_mono_font();
  clicked = clickable_text("time", time_text);
  app_pop_mono_font();

  ImGui::TableSetColumnIndex(1);
  clicked = clickable_text("level", level_label(entry), level_text_color(entry, active)) || clicked;

  ImGui::TableSetColumnIndex(2);
  clicked = clickable_text("source", entry.source) || clicked;

  ImGui::TableSetColumnIndex(3);
  clicked = clickable_text("message", entry.message) || clicked;

  if (clicked) {
    set_tracker_to_log(state, entry);
    state->logs.expanded_index = state->logs.expanded_index == log_index ? -1 : log_index;
  }
  ImGui::PopID();
}

}  // namespace

void draw_logs_tab(AppSession *session, UiState *state) {
  LogsUiState &logs_state = state->logs;
  const RouteData &route_data = session->route_data;
  const RouteLoadSnapshot load = session->route_loader ? session->route_loader->snapshot() : RouteLoadSnapshot{};
  const bool loading_logs = load.active && route_data.logs.empty();
  const std::vector<std::string> sources = collect_log_sources(route_data.logs);

  if (!logs_state.all_sources) {
    logs_state.selected_sources.erase(
      std::remove_if(logs_state.selected_sources.begin(),
                     logs_state.selected_sources.end(),
                     [&](const std::string &source) {
                       return std::find(sources.begin(), sources.end(), source) == sources.end();
                     }),
      logs_state.selected_sources.end());
  }

  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6.0f, 3.0f));
  ImGui::SetNextItemWidth(110.0f);
  const std::string levels_label = level_filter_label(logs_state.enabled_levels_mask);
  if (ImGui::BeginCombo("##logs_level", levels_label.c_str())) {
    bool all_levels = logs_state.enabled_levels_mask == ALL_LEVEL_MASK;
    if (ImGui::Checkbox("All levels", &all_levels)) {
      logs_state.enabled_levels_mask = all_levels ? ALL_LEVEL_MASK : 0u;
    }
    ImGui::Separator();
    for (size_t i = 0; i < LEVEL_OPTIONS.size(); ++i) {
      bool enabled = (logs_state.enabled_levels_mask & (1u << i)) != 0;
      if (ImGui::Checkbox(LEVEL_OPTIONS[i].label, &enabled)) {
        if (enabled) {
          logs_state.enabled_levels_mask |= (1u << i);
        } else {
          logs_state.enabled_levels_mask &= ~(1u << i);
        }
      }
    }
    ImGui::EndCombo();
  }
  ImGui::SameLine();

  ImGui::SetNextItemWidth(150.0f);
  input_text_with_hint_string("##logs_search", "Search...", &logs_state.search);
  ImGui::SameLine();

  const std::string sources_label = source_filter_label(logs_state, sources);
  ImGui::SetNextItemWidth(180.0f);
  if (ImGui::BeginCombo("##logs_source", sources_label.c_str())) {
    bool all_sources = logs_state.all_sources;
    if (ImGui::Checkbox("All sources", &all_sources)) {
      logs_state.all_sources = all_sources;
      if (logs_state.all_sources) {
        logs_state.selected_sources.clear();
      } else {
        logs_state.selected_sources = sources;
      }
    }
    ImGui::Separator();
    for (const std::string &source : sources) {
      bool enabled = logs_state.all_sources
        || std::find(logs_state.selected_sources.begin(), logs_state.selected_sources.end(), source) != logs_state.selected_sources.end();
      if (ImGui::Checkbox(source.c_str(), &enabled)) {
        if (logs_state.all_sources) {
          logs_state.all_sources = false;
          logs_state.selected_sources = sources;
        }
        auto it = std::find(logs_state.selected_sources.begin(), logs_state.selected_sources.end(), source);
        if (enabled) {
          if (it == logs_state.selected_sources.end()) {
            logs_state.selected_sources.push_back(source);
          }
        } else if (it != logs_state.selected_sources.end()) {
          logs_state.selected_sources.erase(it);
        }
        if (logs_state.selected_sources.size() == sources.size()) {
          logs_state.all_sources = true;
          logs_state.selected_sources.clear();
        }
      }
    }
    ImGui::EndCombo();
  }
  ImGui::SameLine();

  ImGui::SetNextItemWidth(110.0f);
  if (ImGui::BeginCombo("##logs_time_mode", time_mode_label(logs_state.time_mode))) {
    for (LogTimeMode mode : {LogTimeMode::Route, LogTimeMode::Boot, LogTimeMode::WallClock}) {
      const bool selected = logs_state.time_mode == mode;
      if (ImGui::Selectable(time_mode_label(mode), selected)) {
        logs_state.time_mode = mode;
      }
    }
    ImGui::EndCombo();
  }

  const std::vector<int> filtered_indices = filter_log_indices(route_data, logs_state);
  const bool have_tracker = state->has_tracker_time && !filtered_indices.empty();
  const int active_pos = have_tracker ? find_active_log_position(route_data, filtered_indices, state->tracker_time) : -1;

  ImGui::SameLine();
  ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(), ImGui::GetWindowContentRegionMax().x - 110.0f));
  ImGui::Text("%zu / %zu", filtered_indices.size(), route_data.logs.size());
  ImGui::PopStyleVar();

  if (route_data.logs.empty()) {
    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_Text, color_rgb(116, 124, 133));
    ImGui::TextWrapped("%s", loading_logs ? "Loading logs..." : "No text logs available for this route.");
    ImGui::PopStyleColor();
    return;
  }

  if (ImGui::BeginChild("##logs_table_child", ImVec2(0.0f, 0.0f), false)) {
    if (have_tracker && std::abs(logs_state.last_auto_scroll_time - state->tracker_time) > 1.0e-6) {
      const float row_height = ImGui::GetTextLineHeightWithSpacing() + 6.0f;
      const float visible_h = std::max(1.0f, ImGui::GetWindowHeight());
      const float target = std::max(0.0f, static_cast<float>(active_pos) * row_height - visible_h * 0.45f);
      ImGui::SetScrollY(target);
      logs_state.last_auto_scroll_time = state->tracker_time;
    }

    if (ImGui::BeginTable("##logs_table",
                          4,
                          ImGuiTableFlags_BordersInnerV |
                            ImGuiTableFlags_RowBg |
                            ImGuiTableFlags_Resizable |
                            ImGuiTableFlags_SizingStretchProp)) {
      ImGui::TableSetupColumn("Time", ImGuiTableColumnFlags_WidthFixed, 120.0f);
      ImGui::TableSetupColumn("Level", ImGuiTableColumnFlags_WidthFixed, 72.0f);
      ImGui::TableSetupColumn("Source", ImGuiTableColumnFlags_WidthFixed, 180.0f);
      ImGui::TableSetupColumn("Message", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableHeadersRow();

      const bool use_clipper = logs_state.expanded_index < 0;
      if (use_clipper) {
        ImGuiListClipper clipper;
        clipper.Begin(static_cast<int>(filtered_indices.size()));
        while (clipper.Step()) {
          for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
            const int log_index = filtered_indices[static_cast<size_t>(i)];
            const LogEntry &entry = route_data.logs[static_cast<size_t>(log_index)];
            draw_log_row(entry, log_index, i == active_pos, state);
          }
        }
      } else {
        for (int i = 0; i < static_cast<int>(filtered_indices.size()); ++i) {
          const int log_index = filtered_indices[static_cast<size_t>(i)];
          const LogEntry &entry = route_data.logs[static_cast<size_t>(log_index)];
          draw_log_row(entry, log_index, i == active_pos, state);
          if (logs_state.expanded_index == log_index) {
            draw_log_expansion_row(entry);
          }
        }
      }

      ImGui::EndTable();
    }
  }
  ImGui::EndChild();
}
