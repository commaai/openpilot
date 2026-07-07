#include "tools/cabana/settings.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "json11/json11.hpp"

namespace {

namespace fs = std::filesystem;

std::string home_dir() {
  const char *home = std::getenv("HOME");
  return home ? std::string(home) : std::string();
}

fs::path settings_dir() {
  if (const char *xdg = std::getenv("XDG_CONFIG_HOME"); xdg && xdg[0] != '\0') {
    return fs::path(xdg) / "cabana";
  }
  return fs::path(home_dir()) / ".config" / "cabana";
}

fs::path settings_path() {
  return settings_dir() / "settings.json";
}

json11::Json::array to_json(const std::vector<std::string> &v) {
  json11::Json::array arr;
  arr.reserve(v.size());
  for (const auto &s : v) arr.push_back(s);
  return arr;
}

std::vector<std::string> from_json(const json11::Json &j) {
  std::vector<std::string> v;
  for (const auto &item : j.array_items()) v.push_back(item.string_value());
  return v;
}

}  // namespace

Settings settings;

Settings::Settings() {
  // json11 keeps lazily-constructed static singletons (e.g. Json(bool) shares
  // one); touching them here nests their construction inside ours so
  // [basic.start.term] destroys them after us, keeping save() safe in our
  // destructor even on a first run where load() below never reaches json11.
  (void)json11::Json(false);

  last_dir = last_route_dir = home_dir();
  log_path = home_dir() + "/cabana_live_stream/";
  load();
}

Settings::~Settings() {
  save();
}

void Settings::load() {
  std::ifstream file(settings_path(), std::ios::binary);
  if (!file.is_open()) return;

  std::ostringstream ss;
  ss << file.rdbuf();
  std::string err;
  json11::Json json = json11::Json::parse(ss.str(), err);
  if (!err.empty() || !json.is_object()) return;

  auto get_bool = [&](const char *key, bool def) {
    const json11::Json &v = json[key];
    return v.is_bool() ? v.bool_value() : def;
  };
  auto get_int = [&](const char *key, int def) {
    const json11::Json &v = json[key];
    return v.is_number() ? v.int_value() : def;
  };
  auto get_string = [&](const char *key, const std::string &def) {
    const json11::Json &v = json[key];
    return v.is_string() ? v.string_value() : def;
  };
  auto get_strings = [&](const char *key, const std::vector<std::string> &def) {
    const json11::Json &v = json[key];
    return v.is_array() ? from_json(v) : def;
  };

  absolute_time = get_bool("absolute_time", absolute_time);
  fps = get_int("fps", fps);
  max_cached_minutes = get_int("max_cached_minutes", max_cached_minutes);
  chart_height = get_int("chart_height", chart_height);
  chart_column_count = get_int("chart_column_count", chart_column_count);
  chart_range = get_int("chart_range", chart_range);
  chart_series_type = get_int("chart_series_type", chart_series_type);
  theme = get_int("theme", theme);
  sparkline_range = get_int("sparkline_range", sparkline_range);
  multiple_lines_hex = get_bool("multiple_lines_hex", multiple_lines_hex);
  log_livestream = get_bool("log_livestream", log_livestream);
  suppress_defined_signals = get_bool("suppress_defined_signals", suppress_defined_signals);
  log_path = get_string("log_path", log_path);
  last_dir = get_string("last_dir", last_dir);
  last_route_dir = get_string("last_route_dir", last_route_dir);
  recent_files = get_strings("recent_files", recent_files);
  drag_direction = (DragDirection)get_int("drag_direction", drag_direction);

  recent_dbc_file = get_string("recent_dbc_file", recent_dbc_file);
  active_msg_id = get_string("active_msg_id", active_msg_id);
  selected_msg_ids = get_strings("selected_msg_ids", selected_msg_ids);
  active_charts = get_strings("active_charts", active_charts);
}

void Settings::save() const {
  json11::Json::object obj = {
    {"absolute_time", absolute_time},
    {"fps", fps},
    {"max_cached_minutes", max_cached_minutes},
    {"chart_height", chart_height},
    {"chart_column_count", chart_column_count},
    {"chart_range", chart_range},
    {"chart_series_type", chart_series_type},
    {"theme", theme},
    {"sparkline_range", sparkline_range},
    {"multiple_lines_hex", multiple_lines_hex},
    {"log_livestream", log_livestream},
    {"suppress_defined_signals", suppress_defined_signals},
    {"log_path", log_path},
    {"last_dir", last_dir},
    {"last_route_dir", last_route_dir},
    {"recent_files", to_json(recent_files)},
    {"drag_direction", (int)drag_direction},
    {"recent_dbc_file", recent_dbc_file},
    {"active_msg_id", active_msg_id},
    {"selected_msg_ids", to_json(selected_msg_ids)},
    {"active_charts", to_json(active_charts)},
  };

  std::error_code ec;
  fs::path dir = settings_dir();
  fs::create_directories(dir, ec);
  if (ec) return;

  fs::path path = settings_path();
  fs::path tmp_path = dir / "settings.json.tmp";
  {
    std::ofstream tmp(tmp_path, std::ios::binary | std::ios::trunc);
    if (!tmp.is_open()) return;
    tmp << json11::Json(obj).dump();
    if (!tmp) return;
  }
  std::rename(tmp_path.c_str(), path.c_str());
}
