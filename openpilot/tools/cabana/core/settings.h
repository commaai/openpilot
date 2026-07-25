#pragma once

#include <string>
#include <vector>

constexpr int LIGHT_THEME = 1;
constexpr int DARK_THEME = 2;

struct CabanaSettingsState {
  enum DragDirection { MsbFirst, LsbFirst, AlwaysLE, AlwaysBE };

  bool absolute_time = false;
  int fps = 10;
  int max_cached_minutes = 30;
  int chart_height = 200;
  int chart_column_count = 1;
  int chart_range = 3 * 60;
  int chart_series_type = 0;
  int theme = 0;
  int sparkline_range = 15;
  bool multiple_lines_hex = false;
  bool log_livestream = true;
  bool suppress_defined_signals = false;
  std::string log_path;
  std::string last_dir;
  std::string last_route_dir;
  std::vector<std::string> recent_files;
  DragDirection drag_direction = MsbFirst;

  std::string recent_dbc_file;
  std::string active_msg_id;
  std::vector<std::string> selected_msg_ids;
  std::vector<std::string> active_charts;
};
