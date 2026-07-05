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
  REQUIRE(loggy::save_loggy_settings(settings, settings_path, &error));
  CHECK(error.empty());

  const loggy::LoggySettingsLoadResult loaded = loggy::load_loggy_settings(settings_path);
  REQUIRE(loaded.loaded);
  CHECK(loaded.error.empty());
  CHECK(loaded.settings.opendbc_root == "/tmp/opendbc/dbc");
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
  CHECK(missing.settings.recent_dbc_files.empty());
  CHECK(missing.settings.dbc_assignments.empty());

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
    "dbc": {
      "recent_files": [123, "/tmp/ok.dbc", "", "/tmp/ok.dbc"],
      "opendbc_root": "/tmp/opendbc",
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
  REQUIRE(parsed.settings.recent_dbc_files.size() == 1);
  CHECK(parsed.settings.recent_dbc_files[0] == "/tmp/ok.dbc");
  REQUIRE(parsed.settings.dbc_assignments.size() == 1);
  CHECK(parsed.settings.dbc_assignments.at("1") == "/tmp/one.dbc");

  loggy::LoggySettings overlong;
  overlong.opendbc_root.assign(loggy::kMaxSettingsValueBytes + 1, 'x');
  loggy::normalize_loggy_settings(&overlong);
  CHECK(overlong.opendbc_root.empty());
}
