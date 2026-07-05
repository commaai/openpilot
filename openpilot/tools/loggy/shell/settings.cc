#include "tools/loggy/shell/settings.h"

#include "json11/json11.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <set>
#include <system_error>
#include <utility>

namespace loggy {
namespace fs = std::filesystem;

namespace {

bool bounded_string(const std::string &value, size_t max_bytes) {
  return !value.empty() && value.size() <= max_bytes;
}

json11::Json::array recent_files_json(const LoggySettings &settings) {
  json11::Json::array recent;
  for (const std::string &path : settings.recent_dbc_files) recent.push_back(path);
  return recent;
}

json11::Json::object assignments_json(const LoggySettings &settings) {
  json11::Json::object assignments;
  for (const auto &[source, path] : settings.dbc_assignments) assignments[source] = path;
  return assignments;
}

void set_error(std::string *error, std::string message) {
  if (error != nullptr) *error = std::move(message);
}

}  // namespace

void normalize_loggy_settings(LoggySettings *settings) {
  if (settings == nullptr) return;
  if (!settings->opendbc_root.empty() && !bounded_string(settings->opendbc_root, kMaxSettingsValueBytes)) {
    settings->opendbc_root.clear();
  }

  std::vector<std::string> recent;
  std::set<std::string> seen;
  for (const std::string &path : settings->recent_dbc_files) {
    if (!bounded_string(path, kMaxSettingsValueBytes)) continue;
    if (!seen.insert(path).second) continue;
    recent.push_back(path);
    if (recent.size() >= kMaxRecentDbcFiles) break;
  }
  settings->recent_dbc_files = std::move(recent);

  for (auto it = settings->dbc_assignments.begin(); it != settings->dbc_assignments.end();) {
    if (!bounded_string(it->first, kMaxSettingsKeyBytes) ||
        !bounded_string(it->second, kMaxSettingsValueBytes)) {
      it = settings->dbc_assignments.erase(it);
    } else {
      ++it;
    }
  }
  while (settings->dbc_assignments.size() > kMaxDbcAssignments) {
    auto it = settings->dbc_assignments.end();
    --it;
    settings->dbc_assignments.erase(it);
  }
}

void remember_recent_dbc_file(LoggySettings *settings, std::string path) {
  if (settings == nullptr || !bounded_string(path, kMaxSettingsValueBytes)) return;
  std::vector<std::string> &recent = settings->recent_dbc_files;
  recent.erase(std::remove(recent.begin(), recent.end(), path), recent.end());
  recent.insert(recent.begin(), std::move(path));
  normalize_loggy_settings(settings);
}

void clear_dbc_assignments_for_path(LoggySettings *settings, std::string_view path) {
  if (settings == nullptr || path.empty()) return;
  for (auto it = settings->dbc_assignments.begin(); it != settings->dbc_assignments.end();) {
    if (it->second == path) {
      it = settings->dbc_assignments.erase(it);
    } else {
      ++it;
    }
  }
}

void set_dbc_assignment(LoggySettings *settings, std::string source, std::string path) {
  if (settings == nullptr || !bounded_string(source, kMaxSettingsKeyBytes)) return;
  if (path.empty()) {
    settings->dbc_assignments.erase(source);
    return;
  }
  if (!bounded_string(path, kMaxSettingsValueBytes)) return;
  settings->dbc_assignments[std::move(source)] = std::move(path);
  normalize_loggy_settings(settings);
}

fs::path default_loggy_settings_path() {
  if (const char *xdg = std::getenv("XDG_CONFIG_HOME"); xdg != nullptr && xdg[0] != '\0') {
    return fs::path(xdg) / "loggy" / "settings.json";
  }
  if (const char *home = std::getenv("HOME"); home != nullptr && home[0] != '\0') {
    return fs::path(home) / ".config" / "loggy" / "settings.json";
  }
  return fs::temp_directory_path() / "loggy" / "settings.json";
}

std::string loggy_settings_to_json(const LoggySettings &settings) {
  LoggySettings normalized = settings;
  normalize_loggy_settings(&normalized);

  const json11::Json root = json11::Json::object{
    {"version", kLoggySettingsVersion},
    {"dbc", json11::Json::object{
      {"opendbc_root", normalized.opendbc_root},
      {"recent_files", recent_files_json(normalized)},
      {"assignments", assignments_json(normalized)},
    }},
  };
  return root.dump() + "\n";
}

LoggySettingsLoadResult loggy_settings_from_json(std::string_view json_text) {
  LoggySettingsLoadResult result;
  std::string parse_error;
  const json11::Json root = json11::Json::parse(std::string(json_text), parse_error);
  if (!parse_error.empty()) {
    result.error = parse_error;
    return result;
  }
  if (!root.is_object()) {
    result.error = "settings JSON root is not an object";
    return result;
  }

  const json11::Json &dbc = root["dbc"].is_object() ? root["dbc"] : root;
  const json11::Json &opendbc_root = dbc["opendbc_root"].is_string() ? dbc["opendbc_root"] : root["opendbc_root"];
  if (opendbc_root.is_string()) result.settings.opendbc_root = opendbc_root.string_value();

  const json11::Json &recent_files = dbc["recent_files"].is_array() ? dbc["recent_files"] : root["recent_dbc_files"];
  for (const json11::Json &path : recent_files.array_items()) {
    if (path.is_string()) result.settings.recent_dbc_files.push_back(path.string_value());
  }

  const json11::Json &assignments = dbc["assignments"].is_object() ? dbc["assignments"] : root["dbc_assignments"];
  for (const auto &[source, path] : assignments.object_items()) {
    if (path.is_string()) result.settings.dbc_assignments[source] = path.string_value();
  }

  normalize_loggy_settings(&result.settings);
  result.loaded = true;
  return result;
}

LoggySettingsLoadResult load_loggy_settings(const fs::path &path) {
  LoggySettingsLoadResult result;
  if (path.empty()) {
    result.error = "settings path is empty";
    return result;
  }

  std::error_code ec;
  const bool exists = fs::exists(path, ec);
  if (ec) {
    result.error = "failed to stat settings file: " + ec.message();
    return result;
  }
  if (!exists) return result;

  std::ifstream stream(path, std::ios::binary);
  if (!stream.is_open()) {
    result.error = "failed to open settings file: " + path.string();
    return result;
  }

  const std::string json_text((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
  if (stream.bad()) {
    result.error = "failed to read settings file: " + path.string();
    return result;
  }

  result = loggy_settings_from_json(json_text);
  return result;
}

bool save_loggy_settings(const LoggySettings &settings, const fs::path &path, std::string *error) {
  if (path.empty()) {
    set_error(error, "settings path is empty");
    return false;
  }

  std::error_code ec;
  const fs::path parent = path.parent_path();
  if (!parent.empty()) {
    fs::create_directories(parent, ec);
    if (ec) {
      set_error(error, "failed to create settings directory: " + ec.message());
      return false;
    }
  }

  const std::string json_text = loggy_settings_to_json(settings);
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream.is_open()) {
    set_error(error, "failed to open settings file for writing: " + path.string());
    return false;
  }
  stream.write(json_text.data(), static_cast<std::streamsize>(json_text.size()));
  if (!stream) {
    set_error(error, "failed to write settings file: " + path.string());
    return false;
  }

  set_error(error, "");
  return true;
}

}  // namespace loggy
