#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "tools/loggy/backend/route.h"

namespace loggy {

struct PaneInstance;
struct Session;

struct LogPaneState {
  std::string filter;
  std::string source_filter;
  uint8_t min_level = 0;
  int origin_filter = -1;
  int time_mode = 0;
  bool follow = true;
  size_t max_rows = 500;
};

struct LogFilterParams {
  std::string filter;
  std::string source_filter;
  uint8_t min_level = 0;
  int origin_filter = -1;
  size_t max_rows = 500;
};

LogPaneState parse_logs_pane_state(std::string_view state_json);
std::string logs_pane_state_json(const LogPaneState &state);
const char *log_origin_label(LogOrigin origin);
const char *log_level_label(uint8_t level);
const char *log_time_mode_label(int mode);
std::string log_time_text(const LogEntry &entry, int mode);
std::vector<size_t> filter_log_entries(const std::vector<LogEntry> &logs,
                                       const LogFilterParams &params);
std::vector<size_t> filter_log_entries(const std::vector<LogEntry> &logs,
                                       std::string_view filter,
                                       uint8_t min_level,
                                       size_t max_rows);

void draw_logs_pane(Session &session, PaneInstance &pane);

}  // namespace loggy
