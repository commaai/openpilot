#pragma once

#include "tools/loggy/backend/dbc/dbcmanager.h"
#include "tools/loggy/shell/pane.h"

#include "json11/json11.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace loggy {

struct DbcPaneState {
  std::string path;
  std::string save_as_path;
  std::string sources = "all";
  std::string opendbc_root;
  std::string opendbc_filter;
  std::string status;
};

struct DbcFileRow {
  std::string name;
  std::string filename;
  std::string sources;
  size_t message_count = 0;
  size_t signal_count = 0;
};

struct OpendbcFileRow {
  std::string name;
  std::string path;
};

inline std::string dbc_trim(std::string_view text) {
  size_t start = 0;
  while (start < text.size() && std::isspace(static_cast<unsigned char>(text[start]))) ++start;
  size_t end = text.size();
  while (end > start && std::isspace(static_cast<unsigned char>(text[end - 1]))) --end;
  return std::string(text.substr(start, end - start));
}

inline DbcPaneState parse_dbc_pane_state(std::string_view state_json) {
  DbcPaneState state;
  std::string err;
  const json11::Json json = json11::Json::parse(std::string(state_json), err);
  if (!err.empty() || !json.is_object()) return state;
  if (json["path"].is_string()) state.path = json["path"].string_value();
  if (json["save_as_path"].is_string()) state.save_as_path = json["save_as_path"].string_value();
  if (json["sources"].is_string()) state.sources = json["sources"].string_value();
  if (json["opendbc_root"].is_string()) state.opendbc_root = json["opendbc_root"].string_value();
  if (json["opendbc_filter"].is_string()) state.opendbc_filter = json["opendbc_filter"].string_value();
  if (json["status"].is_string()) state.status = json["status"].string_value();
  return state;
}

inline std::string dbc_pane_state_json(const DbcPaneState &state) {
  return json11::Json(json11::Json::object{
    {"path", state.path},
    {"save_as_path", state.save_as_path},
    {"sources", state.sources},
    {"opendbc_root", state.opendbc_root},
    {"opendbc_filter", state.opendbc_filter},
    {"status", state.status},
  }).dump();
}

inline bool parse_dbc_source_set(std::string_view text, SourceSet *out, std::string *error = nullptr) {
  return parseSourceSet(text, out, error);
}

inline DBCFile *dbc_file_for_sources(DBCManager &manager, const SourceSet &sources) {
  if (sources == SOURCE_ALL) return manager.findDBCFile(0);
  for (const int source : sources) {
    if (source >= 0 && source <= 255) return manager.findDBCFile(static_cast<uint8_t>(source));
  }
  return nullptr;
}

inline constexpr const char *kEmptyDbcTemplate = R"(VERSION ""
NS_ :
BS_:
BU_: XXX
)";

inline bool create_empty_dbc(DBCManager &manager, const SourceSet &sources, std::string name,
                             std::string *error = nullptr) {
  if (name.empty()) name = "untitled";
  return manager.open(sources, name, kEmptyDbcTemplate, error);
}

inline std::string dbc_clipboard_text_for_sources(DBCManager &manager, const SourceSet &sources,
                                                  std::string *error = nullptr) {
  DBCFile *file = dbc_file_for_sources(manager, sources);
  if (file == nullptr) {
    if (error != nullptr) *error = "no DBC for " + toString(sources);
    return {};
  }
  if (error != nullptr) error->clear();
  return file->generateDBC();
}

inline bool open_dbc_from_clipboard_text(DBCManager &manager, const SourceSet &sources,
                                         std::string name, std::string_view text,
                                         std::string *error = nullptr) {
  const std::string content = dbc_trim(text);
  if (content.empty()) {
    if (error != nullptr) *error = "clipboard is empty";
    return false;
  }
  if (name.empty()) name = "clipboard";
  return manager.open(sources, name, content, error);
}

inline std::string dbc_lower(std::string_view text) {
  std::string out(text);
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}

inline std::filesystem::path default_opendbc_root() {
#ifdef LOGGY_REPO_ROOT
  return std::filesystem::path(LOGGY_REPO_ROOT) / "opendbc_repo" / "opendbc" / "dbc";
#else
  return std::filesystem::path("opendbc_repo") / "opendbc" / "dbc";
#endif
}

inline std::vector<OpendbcFileRow> prepare_opendbc_file_rows(const std::filesystem::path &root,
                                                             std::string_view filter,
                                                             size_t max_rows = 1000,
                                                             std::string *error = nullptr) {
  std::vector<OpendbcFileRow> rows;
  std::error_code ec;
  if (root.empty() || !std::filesystem::exists(root, ec) || !std::filesystem::is_directory(root, ec)) {
    if (error != nullptr) *error = "opendbc root not found: " + root.string();
    return rows;
  }

  const std::string normalized_filter = dbc_lower(dbc_trim(filter));
  std::filesystem::directory_iterator it(root, ec);
  if (ec) {
    if (error != nullptr) *error = "failed to read opendbc root: " + ec.message();
    return rows;
  }

  for (const std::filesystem::directory_entry &entry : it) {
    if (rows.size() >= max_rows) break;
    std::error_code entry_ec;
    if (!entry.is_regular_file(entry_ec)) continue;
    const std::filesystem::path path = entry.path();
    if (path.extension() != ".dbc") continue;
    const std::string name = path.stem().string();
    if (!normalized_filter.empty()) {
      const std::string haystack = dbc_lower(name + " " + path.filename().string());
      if (haystack.find(normalized_filter) == std::string::npos) continue;
    }
    rows.push_back(OpendbcFileRow{.name = name, .path = path.string()});
  }

  std::sort(rows.begin(), rows.end(), [](const OpendbcFileRow &a, const OpendbcFileRow &b) {
    return a.name < b.name;
  });
  if (error != nullptr) error->clear();
  return rows;
}

inline std::vector<DbcFileRow> prepare_dbc_file_rows(DBCManager &manager) {
  std::vector<DbcFileRow> rows;
  for (DBCFile *file : manager.allDBCFiles()) {
    if (file == nullptr) continue;
    DbcFileRow row;
    row.name = file->name();
    row.filename = file->filename;
    row.sources = toString(manager.sources(file));
    row.message_count = file->getMessages().size();
    for (const auto &[_, msg] : file->getMessages()) row.signal_count += msg.getSignals().size();
    rows.push_back(std::move(row));
  }
  std::sort(rows.begin(), rows.end(), [](const DbcFileRow &a, const DbcFileRow &b) {
    return a.name < b.name;
  });
  return rows;
}

void draw_dbc_pane(Session &session, PaneInstance &pane);

}  // namespace loggy
