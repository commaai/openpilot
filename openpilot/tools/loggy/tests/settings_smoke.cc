#include "tools/loggy/shell/settings.h"

#include "catch2/catch.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

namespace {

struct TempDir {
  TempDir() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    path = fs::temp_directory_path() / ("loggy_settings_smoke_" + std::to_string(now));
    fs::create_directories(path);
  }

  ~TempDir() {
    std::error_code ec;
    fs::remove_all(path, ec);
  }

  fs::path path;
};

void write_text(const fs::path &path, const std::string &text) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream << text;
}

}  // namespace

TEST_CASE("Loggy settings round-trip DBC recents and assignments") {
  TempDir temp;
  loggy::LoggySettings settings;
  settings.opendbc_root = "/tmp/opendbc/dbc";
  settings.dbc_override = "honda_civic_touring_2016_can_generated";
  settings.map_cache_root = (temp.path / "map-cache").string();
  settings.theme = "light";
  settings.target_fps = 144;
  settings.show_frame_hud = false;
  settings.natural_map_drag = false;

  for (size_t i = 0; i < loggy::kMaxRecentDbcFiles + 3; ++i) {
    loggy::remember_recent_dbc_file(&settings, "/tmp/recent_" + std::to_string(i) + ".dbc");
  }
  loggy::remember_recent_dbc_file(&settings, "/tmp/recent_10.dbc");
  loggy::set_dbc_assignment(&settings, "all", "/tmp/recent_10.dbc");
  loggy::set_dbc_assignment(&settings, "0", "/tmp/recent_10.dbc");
  loggy::clear_dbc_assignments_for_path(&settings, "/tmp/recent_10.dbc");
  loggy::set_dbc_assignment(&settings, "1", "/tmp/recent_10.dbc");
  loggy::set_dbc_assignment(&settings, "source:1", "/tmp/camera.dbc");

  REQUIRE(settings.recent_dbc_files.size() == loggy::kMaxRecentDbcFiles);
  CHECK(settings.recent_dbc_files.front() == "/tmp/recent_10.dbc");
  CHECK(settings.recent_dbc_files.back() == "/tmp/recent_3.dbc");

  std::string error;
  const fs::path settings_path = temp.path / "nested" / "settings.json";
  REQUIRE(loggy::save_loggy_settings(settings, settings_path, error));
  CHECK(error.empty());

  const loggy::LoggySettingsLoadResult loaded = loggy::load_loggy_settings(settings_path);
  REQUIRE(loaded.loaded);
  CHECK(loaded.error.empty());
  CHECK(loaded.settings.opendbc_root == "/tmp/opendbc/dbc");
  CHECK(loaded.settings.dbc_override == "honda_civic_touring_2016_can_generated");
  CHECK(loaded.settings.map_cache_root == (temp.path / "map-cache").string());
  CHECK(loaded.settings.theme == "light");
  CHECK(loaded.settings.target_fps == 144);
  CHECK_FALSE(loaded.settings.show_frame_hud);
  CHECK_FALSE(loaded.settings.natural_map_drag);
  REQUIRE(loaded.settings.recent_dbc_files.size() == loggy::kMaxRecentDbcFiles);
  CHECK(loaded.settings.recent_dbc_files.front() == "/tmp/recent_10.dbc");
  CHECK(loaded.settings.recent_dbc_files.back() == "/tmp/recent_3.dbc");
  CHECK(loaded.settings.dbc_assignments.count("all") == 0);
  CHECK(loaded.settings.dbc_assignments.count("0") == 0);
  CHECK(loaded.settings.dbc_assignments.at("1") == "/tmp/recent_10.dbc");
  CHECK(loaded.settings.dbc_assignments.at("source:1") == "/tmp/camera.dbc");
}

TEST_CASE("Loggy settings default on missing and malformed files") {
  TempDir temp;

  const loggy::LoggySettingsLoadResult missing = loggy::load_loggy_settings(temp.path / "missing.json");
  CHECK_FALSE(missing.loaded);
  CHECK(missing.error.empty());
  CHECK(missing.settings.target_fps == loggy::kDefaultLoggyTargetFps);
  CHECK(missing.settings.theme == loggy::kDefaultTheme);
  CHECK_FALSE(missing.settings.show_frame_hud);
  CHECK(missing.settings.recent_dbc_files.empty());
  CHECK(missing.settings.dbc_assignments.empty());

  const loggy::LoggySettingsLoadResult directory = loggy::load_loggy_settings(temp.path);
  CHECK_FALSE(directory.loaded);
  CHECK_FALSE(directory.error.empty());
  CHECK(directory.settings.recent_dbc_files.empty());
  CHECK(directory.settings.dbc_assignments.empty());

  const fs::path malformed_path = temp.path / "malformed.json";
  write_text(malformed_path, R"({"dbc": )");
  const loggy::LoggySettingsLoadResult malformed = loggy::load_loggy_settings(malformed_path);
  CHECK_FALSE(malformed.loaded);
  CHECK_FALSE(malformed.error.empty());
  CHECK(malformed.settings.recent_dbc_files.empty());
  CHECK(malformed.settings.dbc_assignments.empty());
}

TEST_CASE("Loggy settings ignore malformed fields while keeping valid values") {
  const loggy::LoggySettingsLoadResult parsed = loggy::loggy_settings_from_json(R"({
    "version": 1,
    "app": {
      "target_fps": 999,
      "show_frame_hud": false,
      "map_cache_root": "/tmp/loggy-map-cache",
      "natural_map_drag": false,
      "theme": "light"
    },
    "dbc": {
      "recent_files": [123, "/tmp/ok.dbc", "", "/tmp/ok.dbc"],
      "opendbc_root": "/tmp/opendbc",
      "override": "toyota_new_mc_pt_generated",
      "assignments": {
        "": "/tmp/empty-key.dbc",
        "0": 123,
        "1": "/tmp/one.dbc"
      }
    }
  })");

  REQUIRE(parsed.loaded);
  CHECK(parsed.error.empty());
  CHECK(parsed.settings.opendbc_root == "/tmp/opendbc");
  CHECK(parsed.settings.dbc_override == "toyota_new_mc_pt_generated");
  CHECK(parsed.settings.map_cache_root == "/tmp/loggy-map-cache");
  CHECK(parsed.settings.theme == "light");
  CHECK(parsed.settings.target_fps == loggy::kMaxLoggyTargetFps);
  CHECK_FALSE(parsed.settings.show_frame_hud);
  CHECK_FALSE(parsed.settings.natural_map_drag);
  REQUIRE(parsed.settings.recent_dbc_files.size() == 1);
  CHECK(parsed.settings.recent_dbc_files[0] == "/tmp/ok.dbc");
  REQUIRE(parsed.settings.dbc_assignments.size() == 1);
  CHECK(parsed.settings.dbc_assignments.at("1") == "/tmp/one.dbc");

  loggy::LoggySettings overlong;
  overlong.opendbc_root.assign(loggy::kMaxSettingsValueBytes + 1, 'x');
  overlong.dbc_override.assign(loggy::kMaxSettingsValueBytes + 1, 'x');
  overlong.map_cache_root.assign(loggy::kMaxSettingsValueBytes + 1, 'x');
  overlong.theme = "unknown";
  overlong.target_fps = -20;
  loggy::normalize_loggy_settings(&overlong);
  CHECK(overlong.opendbc_root.empty());
  CHECK(overlong.dbc_override.empty());
  CHECK(overlong.map_cache_root.empty());
  CHECK(overlong.theme == loggy::kDefaultTheme);
  CHECK(overlong.target_fps == loggy::kMinLoggyTargetFps);

  const loggy::LoggySettingsLoadResult legacy = loggy::loggy_settings_from_json(R"({
    "version": 1,
    "target_fps": 45,
    "show_frame_hud": false,
    "recent_dbc_files": ["/tmp/legacy.dbc"]
  })");
  REQUIRE(legacy.loaded);
  CHECK(legacy.settings.target_fps == 45);
  CHECK_FALSE(legacy.settings.show_frame_hud);
  REQUIRE(legacy.settings.recent_dbc_files.size() == 1);
  CHECK(legacy.settings.recent_dbc_files[0] == "/tmp/legacy.dbc");
}
