#include "tools/loggy/backend/session.h"
#include "tools/loggy/panes/dbc.h"
#include "tools/loggy/shell/native_dialog.h"
#include "tools/loggy/shell/theme.h"
#include "tools/loggy/shell/workspace.h"

#include "json11/json11.hpp"

#include "imgui.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <map>
#include <any>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace loggy {
namespace {

struct DbcPaneState {
  std::string path;
  std::string save_as_path;
  std::string sources = "all";
  std::string dbc_override;
  std::string opendbc_root;
  std::string opendbc_filter;
  std::string status;
};

struct DbcFileRow {
  DBCFile *file = nullptr;
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

std::string dbc_trim(std::string_view text) {
  size_t start_ = 0;
  while (start_ < text.size() && std::isspace(static_cast<unsigned char>(text[start_]))) ++start_;
  size_t end = text.size();
  while (end > start_ && std::isspace(static_cast<unsigned char>(text[end - 1]))) --end;
  return std::string(text.substr(start_, end - start_));
}

DbcPaneState parse_dbc_pane_state(std::string_view state_json) {
  DbcPaneState state;
  std::string err;
  const json11::Json json = json11::Json::parse(std::string(state_json), err);
  if (!err.empty() || !json.is_object()) return state;
  if (json["path"].is_string()) state.path = json["path"].string_value();
  if (json["save_as_path"].is_string()) state.save_as_path = json["save_as_path"].string_value();
  if (json["sources"].is_string()) state.sources = json["sources"].string_value();
  if (json["dbc_override"].is_string()) state.dbc_override = json["dbc_override"].string_value();
  if (json["opendbc_root"].is_string()) state.opendbc_root = json["opendbc_root"].string_value();
  if (json["opendbc_filter"].is_string()) state.opendbc_filter = json["opendbc_filter"].string_value();
  if (json["status"].is_string()) state.status = json["status"].string_value();
  return state;
}

std::string dbc_pane_state_json(const DbcPaneState &state) {
  return json11::Json(json11::Json::object{
    {"path", state.path},
    {"save_as_path", state.save_as_path},
    {"sources", state.sources},
    {"dbc_override", state.dbc_override},
    {"opendbc_root", state.opendbc_root},
    {"opendbc_filter", state.opendbc_filter},
    {"status", state.status},
  }).dump();
}

bool parse_dbc_source_set(std::string_view text, SourceSet &out, std::string &error) {
  return parse_source_set(text, out, error);
}

DBCFile *dbc_file_for_sources(DBCManager &manager, const SourceSet &sources) {
  if (sources == SOURCE_ALL) return manager.find_dbc_file(0);
  for (const int source : sources) {
    if (source >= 0 && source <= 255) return manager.find_dbc_file(static_cast<uint8_t>(source));
  }
  return nullptr;
}

constexpr const char *kEmptyDbcTemplate = R"(VERSION ""
NS_ :
BS_:
BU_: XXX
)";

bool create_empty_dbc(DBCManager &manager, const SourceSet &sources, std::string name, std::string &error) {
  if (name.empty()) name = "untitled";
  return manager.open(sources, name, kEmptyDbcTemplate, error);
}

std::string dbc_clipboard_text_for_sources(DBCManager &manager, const SourceSet &sources, std::string &error) {
  DBCFile *file = dbc_file_for_sources(manager, sources);
  if (file == nullptr) {
    error = "no DBC for " + to_string(sources);
    return {};
  }
  error.clear();
  return file->generate_dbc();
}

bool open_dbc_from_clipboard_text(DBCManager &manager, const SourceSet &sources, std::string name,
                                 std::string_view text, std::string &error) {
  const std::string content = dbc_trim(text);
  if (content.empty()) {
    error = "clipboard is empty";
    return false;
  }
  if (name.empty()) name = "clipboard";
  return manager.open(sources, name, content, error);
}

bool assign_dbc_file_sources(DBCManager &manager, DBCFile *file, std::string_view source_text,
                            SourceSet &assigned_sources, std::string &error) {
  SourceSet sources;
  if (!parse_dbc_source_set(source_text, sources, error)) return false;
  if (!manager.assign_sources(file, sources, error)) return false;
  assigned_sources = sources;
  return true;
}

bool dbc_source_sets_conflict(const SourceSet &a, const SourceSet &b) {
  if (a.empty() || b.empty()) return false;
  if (a == SOURCE_ALL || b == SOURCE_ALL) return a == b;
  for (const int source : a) {
    if (b.count(source) != 0) return true;
  }
  return false;
}

bool dbc_assignment_conflicts_loaded_sources(std::string_view source_key,
                                            const std::vector<SourceSet> &loaded_sources) {
  SourceSet sources;
  std::string error;
  if (!parse_dbc_source_set(source_key, sources, error)) return false;
  for (const SourceSet &loaded : loaded_sources) {
    if (dbc_source_sets_conflict(sources, loaded)) return true;
  }
  return false;
}

void sync_dbc_assignments_from_loaded_files(DBCManager &manager, LoggySettings &settings) {
  std::vector<std::pair<SourceSet, std::string>> loaded_assignments;
  std::vector<SourceSet> loaded_sources;
  std::set<std::string> loaded_paths;
  for (DBCFile *file : manager.all_dbc_files()) {
    if (file == nullptr || file->filename.empty()) continue;
    const SourceSet sources = manager.sources(file);
    if (sources.empty()) continue;
    loaded_assignments.push_back({sources, file->filename});
    loaded_sources.push_back(sources);
    loaded_paths.insert(file->filename);
    remember_recent_dbc_file(&settings, file->filename);
  }

  for (auto it = settings.dbc_assignments.begin(); it != settings.dbc_assignments.end();) {
    if (loaded_paths.count(it->second) != 0 ||
        dbc_assignment_conflicts_loaded_sources(it->first, loaded_sources)) {
      it = settings.dbc_assignments.erase(it);
    } else {
      ++it;
    }
  }
  for (const auto &[sources, path] : loaded_assignments) {
    set_dbc_assignment(&settings, to_string(sources), path);
  }
}

std::string dbc_lower(std::string_view text) {
  std::string out(text);
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}

std::filesystem::path default_opendbc_root() {
#ifdef LOGGY_REPO_ROOT
  return std::filesystem::path(LOGGY_REPO_ROOT) / "opendbc_repo" / "opendbc" / "dbc";
#else
  return std::filesystem::path("opendbc_repo") / "opendbc" / "dbc";
#endif
}

std::vector<OpendbcFileRow> prepare_opendbc_file_rows(const std::filesystem::path &root,
                                                     std::string_view filter,
                                                     size_t max_rows,
                                                     std::string &error) {
  std::vector<OpendbcFileRow> rows;
  std::error_code ec;
  if (root.empty() || !std::filesystem::exists(root, ec) || !std::filesystem::is_directory(root, ec)) {
    error = "opendbc root not found: " + root.string();
    return rows;
  }

  const std::string normalized_filter = dbc_lower(dbc_trim(filter));
  std::filesystem::directory_iterator it(root, ec);
  if (ec) {
    error = "failed to read opendbc root: " + ec.message();
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
  error.clear();
  return rows;
}

std::vector<DbcFileRow> prepare_dbc_file_rows(DBCManager &manager) {
  std::vector<DbcFileRow> rows;
  for (DBCFile *file : manager.all_dbc_files()) {
    if (file == nullptr) continue;
    DbcFileRow row;
    row.file = file;
    row.name = file->name();
    row.filename = file->filename;
    row.sources = to_string(manager.sources(file));
    row.message_count = file->messages().size();
    for (const auto &[_, msg] : file->messages()) row.signal_count += msg.signals().size();
    rows.push_back(std::move(row));
  }
  std::sort(rows.begin(), rows.end(), [](const DbcFileRow &a, const DbcFileRow &b) {
    return a.name < b.name;
  });
  return rows;
}

struct OpendbcBrowserCache {
  std::string root;
  std::string filter;
  std::string error;
  std::vector<OpendbcFileRow> rows;
};

struct DbcSourceAssignCache {
  std::map<DBCFile *, std::string> edits;
  std::map<DBCFile *, std::string> last_sources;
};

struct DbcPaneTransientState {
  DbcPaneState state;
  std::string loaded_json;
  OpendbcBrowserCache opendbc;
  DbcSourceAssignCache assign;
};

DbcPaneTransientState &dbc_pane_transient_state(PaneInstance &pane) {
  if (DbcPaneTransientState *state = std::any_cast<DbcPaneTransientState>(&pane.transient_state)) {
    return *state;
  }
  pane.transient_state = DbcPaneTransientState{};
  return std::any_cast<DbcPaneTransientState &>(pane.transient_state);
}

DbcPaneState &dbc_pane_state(PaneInstance &pane, DbcPaneTransientState &transient) {
  if (transient.loaded_json != pane.state_json) {
    transient.state = parse_dbc_pane_state(pane.state_json);
    transient.loaded_json = pane.state_json;
  }
  return transient.state;
}

void save_state(PaneInstance &pane, DbcPaneTransientState &transient, const DbcPaneState &state) {
  pane.state_json = dbc_pane_state_json(state);
  transient.state = state;
  transient.loaded_json = pane.state_json;
}

std::string sources_for_recent(const LoggySettings &settings, const std::string &path, std::string_view fallback) {
  for (const auto &[source_key, assigned_path] : settings.dbc_assignments) {
    if (assigned_path == path) return source_key;
  }
  return std::string(fallback);
}

bool apply_dialog_path(const std::optional<std::string> &path, std::string &target, std::string &status) {
  if (path.has_value()) {
    target = *path;
    status.clear();
    return true;
  }
  status = "Dialog canceled";
  return true;
}

void remember_loaded_dbc_assignments(Session &session, DbcPaneState &state) {
  sync_dbc_assignments_from_loaded_files(session.dbc, session.settings);
  std::string error;
  if (!session.save_settings(error) && !error.empty()) {
    state.status += " (settings: " + error + ")";
  }
}

bool parse_sources_for_action(const DbcPaneState &state, SourceSet &sources, std::string &status) {
  std::string error;
  if (!parse_dbc_source_set(state.sources, sources, error)) {
    status = "DBC action failed: " + error;
    return false;
  }
  return true;
}

void open_dbc_path_for_sources(Session &session, const SourceSet &sources, const std::string &path, DbcPaneState &state) {
  if (path.empty()) {
    state.status = "Open failed: empty path";
    return;
  }
  std::string error;
  if (session.dbc.open(sources, path, error)) {
    state.path = path;
    state.status = "Opened " + path + " for " + to_string(sources);
    if (state.save_as_path.empty()) state.save_as_path = path;
    remember_loaded_dbc_assignments(session, state);
  } else {
    state.status = "Open failed: " + error;
  }
}

void draw_opendbc_browser(Session &session, DbcPaneState &state, OpendbcBrowserCache &cache, bool &changed) {
  if (state.opendbc_root.empty()) {
    state.opendbc_root = session.settings.opendbc_root.empty()
      ? default_opendbc_root().string()
      : session.settings.opendbc_root;
    changed = true;
  }

  ImGui::Separator();
  push_bold_font();
  ImGui::TextUnformatted("opendbc");
  pop_bold_font();

  ImGui::SetNextItemWidth(std::clamp(ImGui::GetContentRegionAvail().x * 0.42f, 220.0f, 520.0f));
  if (input_text_with_hint("Root", "opendbc_repo/opendbc/dbc", &state.opendbc_root)) {
    changed = true;
  }
  if (ImGui::GetContentRegionAvail().x > 92.0f) ImGui::SameLine();
  if (ImGui::Button("Choose##opendbc_root")) {
    std::string error;
    const std::optional<std::string> path = native_dialog_choose_path(
      NativeDialogType::SelectDirectory, {.title = "Select opendbc root", .path = state.opendbc_root}, error);
    if (apply_dialog_path(path, state.opendbc_root, state.status)) {
      changed = true;
      if (!path.has_value() && !error.empty()) state.status = "Dialog failed: " + error;
    }
  }

  if (ImGui::GetContentRegionAvail().x > 190.0f) ImGui::SameLine();
  ImGui::SetNextItemWidth(180.0f);
  if (input_text_with_hint("Filter", "ford", &state.opendbc_filter)) {
    changed = true;
  }

  if (ImGui::GetContentRegionAvail().x > 88.0f) ImGui::SameLine();
  if (ImGui::Button("Scan")) {
    cache.root = state.opendbc_root;
    cache.filter = state.opendbc_filter;
    cache.rows = prepare_opendbc_file_rows(cache.root, cache.filter, 500, cache.error);
    state.status = cache.error.empty()
      ? "Found " + std::to_string(cache.rows.size()) + " opendbc files"
      : cache.error;
    changed = true;
  }

  if (ImGui::GetContentRegionAvail().x > 112.0f) ImGui::SameLine();
  if (ImGui::Button("Save Root")) {
    session.settings.opendbc_root = state.opendbc_root;
    std::string error;
    if (session.save_settings(error)) {
      state.status = "Saved opendbc root";
    } else {
      state.status = "Save root failed: " + error;
    }
    changed = true;
  }

  if (ImGui::GetContentRegionAvail().x > 88.0f) ImGui::SameLine();
  if (ImGui::Button("Default")) {
    state.opendbc_root = default_opendbc_root().string();
    changed = true;
  }

  const bool cache_matches = cache.root == state.opendbc_root && cache.filter == state.opendbc_filter;
  if (!cache_matches) {
    ImGui::TextDisabled("Scan to refresh opendbc results");
    return;
  }
  if (!cache.error.empty()) {
    ImGui::TextDisabled("%s", cache.error.c_str());
    return;
  }
  if (cache.rows.empty()) {
    ImGui::TextDisabled("No opendbc files matched");
    return;
  }

  constexpr ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
                                    ImGuiTableFlags_BordersOuter | ImGuiTableFlags_SizingFixedFit |
                                    ImGuiTableFlags_ScrollY;
  const float table_height = std::min(220.0f, ImGui::GetTextLineHeightWithSpacing() * (static_cast<float>(cache.rows.size()) + 2.0f));
  if (!ImGui::BeginTable("##loggy_opendbc_files", 3, flags, ImVec2(ImGui::GetContentRegionAvail().x, table_height))) return;
  ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, 220.0f);
  ImGui::TableSetupColumn("File", ImGuiTableColumnFlags_WidthStretch);
  ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthFixed, 100.0f);
  ImGui::TableHeadersRow();
  for (size_t i = 0; i < cache.rows.size(); ++i) {
    const OpendbcFileRow &row = cache.rows[i];
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::TextUnformatted(row.name.c_str());
    ImGui::TableSetColumnIndex(1);
    ImGui::TextUnformatted(row.path.c_str());
    ImGui::TableSetColumnIndex(2);
    ImGui::PushID(static_cast<int>(i));
    if (ImGui::SmallButton("Use")) {
      state.path = row.path;
      state.save_as_path = row.path;
      state.status = "Selected " + row.name;
      changed = true;
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("Open")) {
      SourceSet sources;
      if (parse_sources_for_action(state, sources, state.status)) {
        open_dbc_path_for_sources(session, sources, row.path, state);
      }
      changed = true;
    }
    ImGui::PopID();
  }
  ImGui::EndTable();
}

void draw_auto_dbc_controls(Session &session, DbcPaneState &state, bool &changed) {
  if (state.dbc_override.empty() && !session.manual_dbc_name.empty()) {
    state.dbc_override = session.manual_dbc_name;
    changed = true;
  }

  ImGui::Separator();
  push_bold_font();
  ImGui::TextUnformatted("Route DBC");
  pop_bold_font();
  if (!session.car_fingerprint.empty()) {
    ImGui::TextDisabled("Car %s", session.car_fingerprint.c_str());
  }
  ImGui::TextDisabled("Auto %s | Active %s",
                      session.auto_dbc_name.empty() ? "--" : session.auto_dbc_name.c_str(),
                      session.active_dbc_name.empty() ? "--" : session.active_dbc_name.c_str());
  if (!session.dbc_status.empty()) {
    ImGui::TextDisabled("%s", session.dbc_status.c_str());
  }

  ImGui::SetNextItemWidth(std::clamp(ImGui::GetContentRegionAvail().x * 0.42f, 180.0f, 420.0f));
  if (input_text_with_hint("DBC Override", "dbc name or /path/to.dbc", &state.dbc_override)) {
    changed = true;
  }
  if (ImGui::GetContentRegionAvail().x > 90.0f) ImGui::SameLine();
  if (ImGui::Button("Apply DBC")) {
    std::string error;
    if (session.set_manual_dbc_name(state.dbc_override, error)) {
      state.status = state.dbc_override.empty() ? "DBC auto-detect enabled" : "DBC override applied";
    } else {
      state.status = "DBC override failed: " + error;
    }
    changed = true;
  }
  if (ImGui::GetContentRegionAvail().x > 86.0f) ImGui::SameLine();
  if (ImGui::Button("Auto")) {
    state.dbc_override.clear();
    std::string error;
    if (session.set_manual_dbc_name({}, error)) {
      state.status = "DBC auto-detect enabled";
    } else {
      state.status = error.empty() ? "DBC auto-detect enabled" : "DBC auto failed: " + error;
    }
    changed = true;
  }
}

void draw_dbc_rows(Session &session, DbcPaneState &state, DbcSourceAssignCache &cache, bool &changed) {
  DBCManager &manager = session.dbc;
  const std::vector<DbcFileRow> rows = prepare_dbc_file_rows(manager);
  if (rows.empty()) {
    ImGui::TextDisabled("No DBC files loaded");
    return;
  }

  constexpr ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
                                    ImGuiTableFlags_BordersOuter | ImGuiTableFlags_SizingFixedFit |
                                    ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY;
  if (!ImGui::BeginTable("##loggy_dbc_files", 6, flags, ImGui::GetContentRegionAvail())) return;
  ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, 160.0f);
  ImGui::TableSetupColumn("Sources", ImGuiTableColumnFlags_WidthFixed, 110.0f);
  ImGui::TableSetupColumn("Msgs", ImGuiTableColumnFlags_WidthFixed, 54.0f);
  ImGui::TableSetupColumn("Signals", ImGuiTableColumnFlags_WidthFixed, 66.0f);
  ImGui::TableSetupColumn("File", ImGuiTableColumnFlags_WidthFixed, 360.0f);
  ImGui::TableSetupColumn("Assign", ImGuiTableColumnFlags_WidthFixed, 188.0f);
  ImGui::TableHeadersRow();
  for (const DbcFileRow &row : rows) {
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::TextUnformatted(row.name.c_str());
    ImGui::TableSetColumnIndex(1);
    ImGui::TextUnformatted(row.sources.c_str());
    ImGui::TableSetColumnIndex(2);
    ImGui::Text("%zu", row.message_count);
    ImGui::TableSetColumnIndex(3);
    ImGui::Text("%zu", row.signal_count);
    ImGui::TableSetColumnIndex(4);
    if (row.filename.empty()) ImGui::TextDisabled("--");
    else ImGui::TextUnformatted(row.filename.c_str());
    ImGui::TableSetColumnIndex(5);
    if (row.file != nullptr) {
      std::string &edit = cache.edits[row.file];
      std::string &last_sources = cache.last_sources[row.file];
      if (edit.empty() || last_sources != row.sources) edit = row.sources;
      last_sources = row.sources;

      ImGui::PushID(row.file);
      ImGui::SetNextItemWidth(104.0f);
      if (input_text_with_hint("##sources", "all, 0, 1", &edit)) {
        changed = true;
      }
      ImGui::SameLine();
      if (ImGui::SmallButton("Set")) {
        SourceSet assigned_sources;
        std::string error;
        if (assign_dbc_file_sources(manager, row.file, edit, assigned_sources, error)) {
          edit = to_string(assigned_sources);
          state.sources = edit;
          state.status = "Assigned " + row.name + " to " + edit;
          remember_loaded_dbc_assignments(session, state);
        } else {
          state.status = "Assign failed: " + error;
        }
        changed = true;
      }
      ImGui::PopID();
    }
  }
  ImGui::EndTable();
}

const char *undo_entry_state_label(const UndoStackEntry &entry) {
  if (entry.next) return "next";
  return entry.applied ? "done" : "redo";
}

void draw_dbc_command_history(Session &session) {
  UndoStack &undo = session.dbc_undo;

  ImGui::Separator();
  push_bold_font();
  ImGui::TextUnformatted("DBC Commands");
  pop_bold_font();

  const bool undo_disabled = !undo.can_undo();
  if (undo_disabled) ImGui::BeginDisabled();
  if (ImGui::Button("Undo")) undo.undo();
  if (undo_disabled) ImGui::EndDisabled();
  if (undo.can_undo() && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", undo.undo_text().c_str());

  ImGui::SameLine();
  const bool redo_disabled = !undo.can_redo();
  if (redo_disabled) ImGui::BeginDisabled();
  if (ImGui::Button("Redo")) undo.redo();
  if (redo_disabled) ImGui::EndDisabled();
  if (undo.can_redo() && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", undo.redo_text().c_str());

  ImGui::SameLine();
  ImGui::TextDisabled("%d commands | index %d%s", undo.count(), undo.index(), undo.is_clean() ? " | clean" : "");

  const std::vector<UndoStackEntry> entries = undo.entries();
  if (entries.empty()) {
    ImGui::TextDisabled("No DBC edit commands yet");
    return;
  }

  constexpr ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
                                    ImGuiTableFlags_BordersOuter | ImGuiTableFlags_SizingFixedFit |
                                    ImGuiTableFlags_ScrollY;
  const float table_height = std::min(180.0f, ImGui::GetTextLineHeightWithSpacing() *
                                                (static_cast<float>(entries.size()) + 2.0f));
  if (!ImGui::BeginTable("##loggy_dbc_commands", 4, flags, ImVec2(ImGui::GetContentRegionAvail().x, table_height))) return;
  ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 36.0f);
  ImGui::TableSetupColumn("State", ImGuiTableColumnFlags_WidthFixed, 58.0f);
  ImGui::TableSetupColumn("Command", ImGuiTableColumnFlags_WidthStretch);
  ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthFixed, 54.0f);
  ImGui::TableHeadersRow();
  for (const UndoStackEntry &entry : entries) {
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("%d", entry.index + 1);
    ImGui::TableSetColumnIndex(1);
    if (entry.clean) {
      ImGui::Text("%s *", undo_entry_state_label(entry));
    } else {
      ImGui::TextUnformatted(undo_entry_state_label(entry));
    }
    ImGui::TableSetColumnIndex(2);
    ImGui::TextUnformatted(entry.text.empty() ? "(untitled command)" : entry.text.c_str());
    ImGui::TableSetColumnIndex(3);
    const int target_index = entry.index + 1;
    const bool current = target_index == undo.index();
    if (current) ImGui::BeginDisabled();
    ImGui::PushID(entry.index);
    if (ImGui::SmallButton("Go")) undo.set_index(target_index);
    ImGui::PopID();
    if (current) ImGui::EndDisabled();
  }
  ImGui::EndTable();
}

}  // namespace

void draw_dbc_pane(Session &session, PaneInstance &pane) {
  DbcPaneTransientState &transient_state = dbc_pane_transient_state(pane);
  DBCManager &manager = session.dbc;
  DbcPaneState &state = dbc_pane_state(pane, transient_state);
  bool changed = false;

  ImGui::SetNextItemWidth(std::clamp(ImGui::GetContentRegionAvail().x * 0.42f, 180.0f, 420.0f));
  if (input_text_with_hint("DBC", "/path/to/file.dbc", &state.path)) {
    changed = true;
  }
  if (ImGui::GetContentRegionAvail().x > 92.0f) ImGui::SameLine();
  if (ImGui::Button("Browse##dbc_open")) {
    std::string error;
    const std::optional<std::string> path = native_dialog_choose_path(
      NativeDialogType::OpenFile, {.title = "Open DBC", .path = state.path}, error);
    if (apply_dialog_path(path, state.path, state.status)) {
      if (state.save_as_path.empty()) state.save_as_path = state.path;
      changed = true;
      if (!path.has_value() && !error.empty()) state.status = "Dialog failed: " + error;
    }
  }

  if (ImGui::GetContentRegionAvail().x > 170.0f) ImGui::SameLine();
  ImGui::SetNextItemWidth(120.0f);
  if (input_text_with_hint("Sources", "all, 0, 1", &state.sources)) {
    changed = true;
  }

  const LoggySettings &settings = session.settings;
  if (!settings.recent_dbc_files.empty()) {
    if (ImGui::GetContentRegionAvail().x > 170.0f) ImGui::SameLine();
    ImGui::SetNextItemWidth(std::clamp(ImGui::GetContentRegionAvail().x * 0.36f, 180.0f, 360.0f));
    const char *preview = state.path.empty() ? "Recent DBCs" : state.path.c_str();
    if (ImGui::BeginCombo("Recent", preview)) {
      for (const std::string &path : settings.recent_dbc_files) {
        const bool selected = path == state.path;
        if (ImGui::Selectable(path.c_str(), selected)) {
          state.path = path;
          if (state.save_as_path.empty()) state.save_as_path = path;
          state.sources = sources_for_recent(settings, path, state.sources);
          state.status = "Selected recent DBC";
          changed = true;
        }
        if (selected) ImGui::SetItemDefaultFocus();
      }
      ImGui::EndCombo();
    }
  }

  SourceSet sources;
  if (ImGui::Button("New")) {
    if (parse_sources_for_action(state, sources, state.status)) {
      std::string error;
      if (create_empty_dbc(manager, sources, "untitled", error)) {
        state.status = "Created untitled DBC for " + to_string(sources);
      } else {
        state.status = "New failed: " + error;
      }
    }
    changed = true;
  }

  if (ImGui::GetContentRegionAvail().x > 88.0f) ImGui::SameLine();
  if (ImGui::Button("Open")) {
    if (state.path.empty()) {
      state.status = "Open failed: empty path";
    } else if (parse_sources_for_action(state, sources, state.status)) {
      open_dbc_path_for_sources(session, sources, state.path, state);
    }
    changed = true;
  }

  if (ImGui::GetContentRegionAvail().x > 88.0f) ImGui::SameLine();
  if (ImGui::Button("Save")) {
    if (parse_sources_for_action(state, sources, state.status)) {
      DBCFile *file = dbc_file_for_sources(manager, sources);
      if (file == nullptr) {
        state.status = "Save failed: no DBC for " + to_string(sources);
      } else if (file->filename.empty()) {
        state.status = "Save failed: use Save As for untitled DBC";
      } else {
        state.status = file->save() ? "Saved " + file->filename : "Save failed: " + file->filename;
      }
    }
    changed = true;
  }

  if (ImGui::GetContentRegionAvail().x > 118.0f) ImGui::SameLine();
  ImGui::SetNextItemWidth(std::clamp(ImGui::GetContentRegionAvail().x * 0.38f, 160.0f, 360.0f));
  if (input_text_with_hint("Save As", "/tmp/loggy.dbc", &state.save_as_path)) {
    changed = true;
  }
  if (ImGui::GetContentRegionAvail().x > 92.0f) ImGui::SameLine();
  if (ImGui::Button("Choose##dbc_save_as")) {
    std::string error;
    const std::optional<std::string> path = native_dialog_choose_path(
      NativeDialogType::SaveFile,
      {.title = "Save DBC As", .path = state.save_as_path, .confirm_overwrite = true},
      error);
    if (apply_dialog_path(path, state.save_as_path, state.status)) {
      changed = true;
      if (!path.has_value() && !error.empty()) state.status = "Dialog failed: " + error;
    }
  }

  if (ImGui::GetContentRegionAvail().x > 108.0f) ImGui::SameLine();
  if (ImGui::Button("Save As")) {
    if (state.save_as_path.empty()) {
      state.status = "Save As failed: empty path";
    } else if (parse_sources_for_action(state, sources, state.status)) {
      DBCFile *file = dbc_file_for_sources(manager, sources);
      if (file == nullptr) {
        state.status = "Save As failed: no DBC for " + to_string(sources);
      } else {
        if (file->save_as(state.save_as_path)) {
          state.status = "Saved " + state.save_as_path;
          remember_loaded_dbc_assignments(session, state);
        } else {
          state.status = "Save As failed: " + state.save_as_path;
        }
      }
    }
    changed = true;
  }

  if (ImGui::GetContentRegionAvail().x > 92.0f) ImGui::SameLine();
  if (ImGui::Button("Close")) {
    if (parse_sources_for_action(state, sources, state.status)) {
      manager.close(sources);
      transient_state.assign.edits.clear();
      transient_state.assign.last_sources.clear();
      state.status = "Closed " + to_string(sources);
    }
    changed = true;
  }

  if (ImGui::GetContentRegionAvail().x > 96.0f) ImGui::SameLine();
  if (ImGui::Button("Close All")) {
    manager.close_all();
    transient_state.assign.edits.clear();
    transient_state.assign.last_sources.clear();
    state.status = "Closed all DBC files";
    changed = true;
  }

  if (ImGui::GetContentRegionAvail().x > 86.0f) ImGui::SameLine();
  if (ImGui::Button("Copy")) {
    if (parse_sources_for_action(state, sources, state.status)) {
      std::string error;
      const std::string content = dbc_clipboard_text_for_sources(manager, sources, error);
      if (content.empty() && !error.empty()) {
        state.status = "Copy failed: " + error;
      } else {
        ImGui::SetClipboardText(content.c_str());
        state.status = "Copied DBC for " + to_string(sources);
      }
    }
    changed = true;
  }

  if (ImGui::GetContentRegionAvail().x > 88.0f) ImGui::SameLine();
  if (ImGui::Button("Paste")) {
    if (parse_sources_for_action(state, sources, state.status)) {
      const char *clipboard = ImGui::GetClipboardText();
      std::string error;
      if (open_dbc_from_clipboard_text(manager, sources, "clipboard", clipboard == nullptr ? "" : clipboard, error)) {
        state.status = "Pasted DBC for " + to_string(sources);
      } else {
        state.status = "Paste failed: " + error;
      }
    }
    changed = true;
  }

  if (!state.status.empty()) {
    ImGui::SameLine();
    ImGui::TextDisabled("%s", state.status.c_str());
  }
  if (!session.settings_status.empty()) {
    ImGui::TextDisabled("%s", session.settings_status.c_str());
  }

  if (changed) save_state(pane, transient_state, state);

  draw_auto_dbc_controls(session, state, changed);
  if (changed) save_state(pane, transient_state, state);

  draw_opendbc_browser(session, state, transient_state.opendbc, changed);
  if (changed) save_state(pane, transient_state, state);

  draw_dbc_command_history(session);

  ImGui::Separator();
  push_bold_font();
  ImGui::TextUnformatted("Loaded DBCs");
  pop_bold_font();
  draw_dbc_rows(session, state, transient_state.assign, changed);
  if (changed) save_state(pane, transient_state, state);
}

}  // namespace loggy
