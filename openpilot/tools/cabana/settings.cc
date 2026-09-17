#include "tools/cabana/settings.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#endif

#include <type_traits>

#include "json11/json11.hpp"
#include "tools/cabana/utils/util.h"

Settings settings;

namespace {

std::filesystem::path settingsFile() {
  return utils::configPath() / "cabana.json";
}

struct LoadedSettings {
  json11::Json::object values;
  bool exists = false;
  bool valid = true;
};

class FileLock {
public:
  explicit FileLock(const std::filesystem::path &path) {
    fd = open(path.c_str(), O_CREAT | O_CLOEXEC, 0600);
    if (fd < 0 || flock(fd, LOCK_EX) < 0) {
      fprintf(stderr, "failed to lock Cabana settings %s: %s\n", path.c_str(), strerror(errno));
      if (fd >= 0) close(fd);
      fd = -1;
    }
  }
  ~FileLock() {
    if (fd >= 0) close(fd);
  }
  bool isLocked() const { return fd >= 0; }

private:
  int fd = -1;
};

LoadedSettings loadSettings() {
  std::ifstream input(settingsFile());
  if (!input) return {};

  const std::string contents{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  std::string error;
  auto settings_json = json11::Json::parse(contents, error);
  if (!error.empty() || !settings_json.is_object()) {
    fprintf(stderr, "failed to read Cabana settings %s%s%s\n", settingsFile().c_str(), error.empty() ? "" : ": ", error.c_str());
    return {.exists = true, .valid = false};
  }
  return {.values = settings_json.object_items(), .exists = true};
}

bool ensureSettingsDirectory() {
  const auto path = settingsFile();
  std::error_code error;
  std::filesystem::create_directories(path.parent_path(), error);
  if (error) {
    fprintf(stderr, "failed to create Cabana settings directory %s: %s\n", path.parent_path().c_str(), error.message().c_str());
    return false;
  }
  return true;
}

bool writeAll(int fd, const std::string &data) {
  size_t written = 0;
  while (written < data.size()) {
    ssize_t result = write(fd, data.data() + written, data.size() - written);
    if (result < 0 && errno == EINTR) continue;
    if (result <= 0) return false;
    written += result;
  }
  return true;
}

bool saveSettings(const json11::Json::object &settings_json) {
  const auto path = settingsFile();
  const std::string contents = json11::Json(settings_json).dump();
  std::string temporary_path = path.string() + ".tmp.XXXXXX";
  int fd = mkstemp(temporary_path.data());
  if (fd < 0) {
    fprintf(stderr, "failed to create temporary Cabana settings %s: %s\n", temporary_path.c_str(), strerror(errno));
    return false;
  }

  bool success = writeAll(fd, contents) && fsync(fd) == 0;
  if (close(fd) < 0) success = false;
  if (success && rename(temporary_path.c_str(), path.c_str()) < 0) success = false;

  if (success) {
    int dir_fd = open(path.parent_path().c_str(), O_RDONLY | O_CLOEXEC);
    success = dir_fd >= 0 && fsync(dir_fd) == 0;
    if (dir_fd >= 0 && close(dir_fd) < 0) success = false;
  }

  if (!success) {
    const int saved_errno = errno;
    unlink(temporary_path.c_str());
    fprintf(stderr, "failed to save Cabana settings to %s: %s\n", path.c_str(), strerror(saved_errno));
  }
  return success;
}

bool preserveCorruptSettings() {
  const auto path = settingsFile();
  auto backup = path;
  backup += ".corrupt";
  for (int i = 1; std::filesystem::exists(backup); ++i) {
    backup = path;
    backup += ".corrupt." + std::to_string(i);
  }
  if (rename(path.c_str(), backup.c_str()) < 0) {
    fprintf(stderr, "failed to preserve corrupt Cabana settings %s: %s\n", path.c_str(), strerror(errno));
    return false;
  }
  fprintf(stderr, "preserved corrupt Cabana settings at %s\n", backup.c_str());
  return true;
}

template <typename T>
void readSetting(const json11::Json::object &settings_json, const char *key, T &value) {
  auto it = settings_json.find(key);
  if (it == settings_json.end()) return;

  if constexpr (std::is_same_v<T, bool>) {
    if (it->second.is_bool()) value = it->second.bool_value();
  } else if constexpr (std::is_integral_v<T>) {
    if (it->second.is_number()) value = it->second.int_value();
  } else if constexpr (std::is_enum_v<T>) {
    if (it->second.is_number()) value = static_cast<T>(it->second.int_value());
  }
}

void readSetting(const json11::Json::object &settings_json, const char *key, std::string &value) {
  auto it = settings_json.find(key);
  if (it != settings_json.end() && it->second.is_string()) value = it->second.string_value();
}

void readSetting(const json11::Json::object &settings_json, const char *key, std::vector<std::string> &value) {
  auto it = settings_json.find(key);
  if (it == settings_json.end() || !it->second.is_array()) return;

  std::vector<std::string> stored;
  for (const auto &item : it->second.array_items()) {
    if (!item.is_string()) return;
    stored.push_back(item.string_value());
  }
  value = std::move(stored);
}

template <typename T>
void writeSetting(json11::Json::object &settings_json, const char *key, const T &value) {
  if constexpr (std::is_same_v<T, bool>) {
    settings_json[key] = value;
  } else if constexpr (std::is_integral_v<T> || std::is_enum_v<T>) {
    settings_json[key] = static_cast<int>(value);
  }
}

void writeSetting(json11::Json::object &settings_json, const char *key, const std::string &value) {
  settings_json[key] = value;
}

void writeSetting(json11::Json::object &settings_json, const char *key, const std::vector<std::string> &value) {
  settings_json[key] = value;
}

template <class Store, class SettingOperation>
void settingsOp(Store &s, SettingOperation op) {
  op(s, "absolute_time", settings.absolute_time);
  op(s, "max_cached_minutes", settings.max_cached_minutes);
  op(s, "chart_height", settings.chart_height);
  op(s, "chart_range", settings.chart_range);
  op(s, "chart_column_count", settings.chart_column_count);
  op(s, "last_dir", settings.last_dir);
  op(s, "last_route_dir", settings.last_route_dir);
  op(s, "recent_files", settings.recent_files);
  op(s, "ui_state", settings.ui_state);
  op(s, "chart_series_type", settings.chart_series_type);
  op(s, "theme", settings.theme);
  op(s, "sparkline_range", settings.sparkline_range);
  op(s, "multiple_lines_hex", settings.multiple_lines_hex);
  op(s, "log_livestream", settings.log_livestream);
  op(s, "log_path", settings.log_path);
  op(s, "drag_direction", (int &)settings.drag_direction);
  op(s, "suppress_defined_signals", settings.suppress_defined_signals);
  op(s, "crop_video", settings.crop_video);
  op(s, "recent_dbc_file", settings.recent_dbc_file);
  op(s, "active_msg_id", settings.active_msg_id);
  op(s, "selected_msg_ids", settings.selected_msg_ids);
  op(s, "active_charts", settings.active_charts);
}

}  // namespace

Settings::Settings() {
  last_dir = last_route_dir = utils::homePath();
  log_path = utils::homePath() + "/cabana_live_stream/";
  const auto stored_settings = loadSettings();
  if (stored_settings.valid) {
    if (stored_settings.exists) {
      settingsOp(stored_settings.values, [](const auto &s, const char *key, auto &value) { readSetting(s, key, value); });
    }
  }
  // settings written before the "Automatic" theme was dropped hold a 0 for it
  if (theme != LIGHT_THEME && theme != DARK_THEME) theme = LIGHT_THEME;
}

// Must be called before main() returns: json11's internal statistics are constructed on first
// use at runtime, so they are destroyed before this pre-main global. Saving from ~Settings
// would use them after destruction and corrupt the heap.
void Settings::save() {
  if (!ensureSettingsDirectory()) return;

  auto lock_path = settingsFile();
  lock_path += ".lock";
  FileLock lock(lock_path);
  if (!lock.isLocked()) return;

  auto stored_settings = loadSettings();
  if (!stored_settings.valid && !preserveCorruptSettings()) return;
  settingsOp(stored_settings.values, [](auto &s, const char *key, const auto &value) { writeSetting(s, key, value); });
  saveSettings(stored_settings.values);
}
