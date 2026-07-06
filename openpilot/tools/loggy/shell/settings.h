#pragma once

#include <cstddef>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace loggy {

inline constexpr int kLoggySettingsVersion = 1;
inline constexpr int kDefaultLoggyTargetFps = 60;
inline constexpr int kMinLoggyTargetFps = 15;
inline constexpr int kMaxLoggyTargetFps = 240;
inline constexpr const char *kDefaultTheme = "light";
inline constexpr size_t kMaxRecentDbcFiles = 16;
inline constexpr size_t kMaxDbcAssignments = 64;
inline constexpr size_t kMaxSettingsKeyBytes = 128;
inline constexpr size_t kMaxSettingsValueBytes = 4096;

struct LoggySettings {
  std::string opendbc_root;
  std::string dbc_override;
  std::string map_cache_root;
  std::string theme = kDefaultTheme;
  std::vector<std::string> recent_dbc_files;
  int target_fps = kDefaultLoggyTargetFps;
  bool show_frame_hud = false;
  bool natural_map_drag = true;
  // Keys are caller-owned bus/source identifiers such as "all", "0", or
  // future route-specific source labels. Values are DBC file paths.
  std::map<std::string, std::string> dbc_assignments;
};

struct LoggySettingsLoadResult {
  LoggySettings settings;
  bool loaded = false;
  std::string error;
};

void normalize_loggy_settings(LoggySettings *settings);
void remember_recent_dbc_file(LoggySettings *settings, std::string path);
void clear_dbc_assignments_for_path(LoggySettings *settings, std::string_view path);
void set_dbc_assignment(LoggySettings *settings, std::string source, std::string path);

std::filesystem::path default_loggy_settings_path();
std::string loggy_settings_to_json(const LoggySettings &settings);
LoggySettingsLoadResult loggy_settings_from_json(std::string_view json_text);
LoggySettingsLoadResult load_loggy_settings(const std::filesystem::path &path);
bool save_loggy_settings(const LoggySettings &settings, const std::filesystem::path &path,
                         std::string &error);

}  // namespace loggy
