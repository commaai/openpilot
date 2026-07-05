#include "tools/loggy/backend/session.h"
#include "tools/loggy/panes/dbc.h"
#include "tools/loggy/shell/theme.h"
#include "tools/loggy/shell/workspace.h"

#include "imgui.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace loggy {
namespace {

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

void save_state(PaneInstance *pane, const DbcPaneState &state) {
  pane->state_json = dbc_pane_state_json(state);
}

std::string sources_for_recent(const LoggySettings &settings, const std::string &path, std::string_view fallback) {
  for (const auto &[source_key, assigned_path] : settings.dbc_assignments) {
    if (assigned_path == path) return source_key;
  }
  return std::string(fallback);
}

void remember_loaded_dbc_assignments(Session &session, DbcPaneState *state) {
  if (state == nullptr) return;
  sync_dbc_assignments_from_loaded_files(*dbc(), &session.settings());
  std::string error;
  if (!session.saveSettings(&error) && !error.empty()) {
    state->status += " (settings: " + error + ")";
  }
}

bool parse_sources_for_action(const DbcPaneState &state, SourceSet *sources, std::string *status) {
  std::string error;
  if (!parse_dbc_source_set(state.sources, sources, &error)) {
    if (status != nullptr) *status = "DBC action failed: " + error;
    return false;
  }
  return true;
}

void open_dbc_path_for_sources(Session &session, const SourceSet &sources, const std::string &path, DbcPaneState *state) {
  if (state == nullptr) return;
  if (path.empty()) {
    state->status = "Open failed: empty path";
    return;
  }
  std::string error;
  if (dbc()->open(sources, path, &error)) {
    state->path = path;
    state->status = "Opened " + path + " for " + toString(sources);
    if (state->save_as_path.empty()) state->save_as_path = path;
    remember_loaded_dbc_assignments(session, state);
  } else {
    state->status = "Open failed: " + error;
  }
}

void draw_opendbc_browser(Session &session, DbcPaneState *state, OpendbcBrowserCache *cache, bool *changed) {
  if (state == nullptr || cache == nullptr || changed == nullptr) return;
  if (state->opendbc_root.empty()) {
    state->opendbc_root = session.settings().opendbc_root.empty()
      ? default_opendbc_root().string()
      : session.settings().opendbc_root;
    *changed = true;
  }

  ImGui::Separator();
  push_bold_font();
  ImGui::TextUnformatted("opendbc");
  pop_bold_font();

  std::array<char, 256> root_buf{};
  std::snprintf(root_buf.data(), root_buf.size(), "%s", state->opendbc_root.c_str());
  ImGui::SetNextItemWidth(std::clamp(ImGui::GetContentRegionAvail().x * 0.42f, 220.0f, 520.0f));
  if (ImGui::InputTextWithHint("Root", "opendbc_repo/opendbc/dbc", root_buf.data(), root_buf.size())) {
    state->opendbc_root = root_buf.data();
    *changed = true;
  }

  if (ImGui::GetContentRegionAvail().x > 190.0f) ImGui::SameLine();
  std::array<char, 96> filter_buf{};
  std::snprintf(filter_buf.data(), filter_buf.size(), "%s", state->opendbc_filter.c_str());
  ImGui::SetNextItemWidth(180.0f);
  if (ImGui::InputTextWithHint("Filter", "ford", filter_buf.data(), filter_buf.size())) {
    state->opendbc_filter = filter_buf.data();
    *changed = true;
  }

  if (ImGui::GetContentRegionAvail().x > 88.0f) ImGui::SameLine();
  if (ImGui::Button("Scan")) {
    cache->root = state->opendbc_root;
    cache->filter = state->opendbc_filter;
    cache->rows = prepare_opendbc_file_rows(cache->root, cache->filter, 500, &cache->error);
    state->status = cache->error.empty()
      ? "Found " + std::to_string(cache->rows.size()) + " opendbc files"
      : cache->error;
    *changed = true;
  }

  if (ImGui::GetContentRegionAvail().x > 112.0f) ImGui::SameLine();
  if (ImGui::Button("Save Root")) {
    session.settings().opendbc_root = state->opendbc_root;
    std::string error;
    if (session.saveSettings(&error)) {
      state->status = "Saved opendbc root";
    } else {
      state->status = "Save root failed: " + error;
    }
    *changed = true;
  }

  if (ImGui::GetContentRegionAvail().x > 88.0f) ImGui::SameLine();
  if (ImGui::Button("Default")) {
    state->opendbc_root = default_opendbc_root().string();
    *changed = true;
  }

  const bool cache_matches = cache->root == state->opendbc_root && cache->filter == state->opendbc_filter;
  if (!cache_matches) {
    ImGui::TextDisabled("Scan to refresh opendbc results");
    return;
  }
  if (!cache->error.empty()) {
    ImGui::TextDisabled("%s", cache->error.c_str());
    return;
  }
  if (cache->rows.empty()) {
    ImGui::TextDisabled("No opendbc files matched");
    return;
  }

  constexpr ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
                                    ImGuiTableFlags_BordersOuter | ImGuiTableFlags_SizingFixedFit |
                                    ImGuiTableFlags_ScrollY;
  const float table_height = std::min(220.0f, ImGui::GetTextLineHeightWithSpacing() * (static_cast<float>(cache->rows.size()) + 2.0f));
  if (!ImGui::BeginTable("##loggy_opendbc_files", 3, flags, ImVec2(ImGui::GetContentRegionAvail().x, table_height))) return;
  ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, 220.0f);
  ImGui::TableSetupColumn("File", ImGuiTableColumnFlags_WidthStretch);
  ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthFixed, 100.0f);
  ImGui::TableHeadersRow();
  for (size_t i = 0; i < cache->rows.size(); ++i) {
    const OpendbcFileRow &row = cache->rows[i];
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::TextUnformatted(row.name.c_str());
    ImGui::TableSetColumnIndex(1);
    ImGui::TextUnformatted(row.path.c_str());
    ImGui::TableSetColumnIndex(2);
    ImGui::PushID(static_cast<int>(i));
    if (ImGui::SmallButton("Use")) {
      state->path = row.path;
      state->save_as_path = row.path;
      state->status = "Selected " + row.name;
      *changed = true;
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("Open")) {
      SourceSet sources;
      if (parse_sources_for_action(*state, &sources, &state->status)) {
        open_dbc_path_for_sources(session, sources, row.path, state);
      }
      *changed = true;
    }
    ImGui::PopID();
  }
  ImGui::EndTable();
}

void draw_dbc_rows(Session &session, DbcPaneState *state, DbcSourceAssignCache *cache, bool *changed) {
  DBCManager &manager = *dbc();
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
    if (cache != nullptr && row.file != nullptr) {
      std::string &edit = cache->edits[row.file];
      std::string &last_sources = cache->last_sources[row.file];
      if (edit.empty() || last_sources != row.sources) edit = row.sources;
      last_sources = row.sources;

      std::array<char, 64> assign_buf{};
      std::snprintf(assign_buf.data(), assign_buf.size(), "%s", edit.c_str());
      ImGui::PushID(row.file);
      ImGui::SetNextItemWidth(104.0f);
      if (ImGui::InputTextWithHint("##sources", "all, 0, 1", assign_buf.data(), assign_buf.size())) {
        edit = assign_buf.data();
        if (changed != nullptr) *changed = true;
      }
      ImGui::SameLine();
      if (ImGui::SmallButton("Set")) {
        SourceSet assigned_sources;
        std::string error;
        if (assign_dbc_file_sources(manager, row.file, edit, &assigned_sources, &error)) {
          edit = toString(assigned_sources);
          if (state != nullptr) {
            state->sources = edit;
            state->status = "Assigned " + row.name + " to " + edit;
          }
          remember_loaded_dbc_assignments(session, state);
        } else if (state != nullptr) {
          state->status = "Assign failed: " + error;
        }
        if (changed != nullptr) *changed = true;
      }
      ImGui::PopID();
    }
  }
  ImGui::EndTable();
}

}  // namespace

void draw_dbc_pane(Session &session, PaneInstance &pane) {
  static OpendbcBrowserCache opendbc_cache;
  static DbcSourceAssignCache assign_cache;
  DbcPaneState state = parse_dbc_pane_state(pane.state_json);
  bool changed = false;

  std::array<char, 256> path_buf{};
  std::snprintf(path_buf.data(), path_buf.size(), "%s", state.path.c_str());
  ImGui::SetNextItemWidth(std::clamp(ImGui::GetContentRegionAvail().x * 0.42f, 180.0f, 420.0f));
  if (ImGui::InputTextWithHint("DBC", "/path/to/file.dbc", path_buf.data(), path_buf.size())) {
    state.path = path_buf.data();
    changed = true;
  }

  if (ImGui::GetContentRegionAvail().x > 170.0f) ImGui::SameLine();
  std::array<char, 64> source_buf{};
  std::snprintf(source_buf.data(), source_buf.size(), "%s", state.sources.c_str());
  ImGui::SetNextItemWidth(120.0f);
  if (ImGui::InputTextWithHint("Sources", "all, 0, 1", source_buf.data(), source_buf.size())) {
    state.sources = source_buf.data();
    changed = true;
  }

  const LoggySettings &settings = session.settings();
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
    if (parse_sources_for_action(state, &sources, &state.status)) {
      std::string error;
      if (create_empty_dbc(*dbc(), sources, "untitled", &error)) {
        state.status = "Created untitled DBC for " + toString(sources);
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
    } else if (parse_sources_for_action(state, &sources, &state.status)) {
      open_dbc_path_for_sources(session, sources, state.path, &state);
    }
    changed = true;
  }

  if (ImGui::GetContentRegionAvail().x > 88.0f) ImGui::SameLine();
  if (ImGui::Button("Save")) {
    if (parse_sources_for_action(state, &sources, &state.status)) {
      DBCFile *file = dbc_file_for_sources(*dbc(), sources);
      if (file == nullptr) {
        state.status = "Save failed: no DBC for " + toString(sources);
      } else if (file->filename.empty()) {
        state.status = "Save failed: use Save As for untitled DBC";
      } else {
        state.status = file->save() ? "Saved " + file->filename : "Save failed: " + file->filename;
      }
    }
    changed = true;
  }

  if (ImGui::GetContentRegionAvail().x > 118.0f) ImGui::SameLine();
  std::array<char, 256> save_as_buf{};
  std::snprintf(save_as_buf.data(), save_as_buf.size(), "%s", state.save_as_path.c_str());
  ImGui::SetNextItemWidth(std::clamp(ImGui::GetContentRegionAvail().x * 0.38f, 160.0f, 360.0f));
  if (ImGui::InputTextWithHint("Save As", "/tmp/loggy.dbc", save_as_buf.data(), save_as_buf.size())) {
    state.save_as_path = save_as_buf.data();
    changed = true;
  }

  if (ImGui::GetContentRegionAvail().x > 108.0f) ImGui::SameLine();
  if (ImGui::Button("Save As")) {
    if (state.save_as_path.empty()) {
      state.status = "Save As failed: empty path";
    } else if (parse_sources_for_action(state, &sources, &state.status)) {
      DBCFile *file = dbc_file_for_sources(*dbc(), sources);
      if (file == nullptr) {
        state.status = "Save As failed: no DBC for " + toString(sources);
      } else {
        if (file->saveAs(state.save_as_path)) {
          state.status = "Saved " + state.save_as_path;
          remember_loaded_dbc_assignments(session, &state);
        } else {
          state.status = "Save As failed: " + state.save_as_path;
        }
      }
    }
    changed = true;
  }

  if (ImGui::GetContentRegionAvail().x > 92.0f) ImGui::SameLine();
  if (ImGui::Button("Close")) {
    if (parse_sources_for_action(state, &sources, &state.status)) {
      dbc()->close(sources);
      assign_cache.edits.clear();
      assign_cache.last_sources.clear();
      state.status = "Closed " + toString(sources);
    }
    changed = true;
  }

  if (ImGui::GetContentRegionAvail().x > 96.0f) ImGui::SameLine();
  if (ImGui::Button("Close All")) {
    dbc()->closeAll();
    assign_cache.edits.clear();
    assign_cache.last_sources.clear();
    state.status = "Closed all DBC files";
    changed = true;
  }

  if (ImGui::GetContentRegionAvail().x > 86.0f) ImGui::SameLine();
  if (ImGui::Button("Copy")) {
    if (parse_sources_for_action(state, &sources, &state.status)) {
      std::string error;
      const std::string content = dbc_clipboard_text_for_sources(*dbc(), sources, &error);
      if (content.empty() && !error.empty()) {
        state.status = "Copy failed: " + error;
      } else {
        ImGui::SetClipboardText(content.c_str());
        state.status = "Copied DBC for " + toString(sources);
      }
    }
    changed = true;
  }

  if (ImGui::GetContentRegionAvail().x > 88.0f) ImGui::SameLine();
  if (ImGui::Button("Paste")) {
    if (parse_sources_for_action(state, &sources, &state.status)) {
      const char *clipboard = ImGui::GetClipboardText();
      std::string error;
      if (open_dbc_from_clipboard_text(*dbc(), sources, "clipboard", clipboard == nullptr ? "" : clipboard, &error)) {
        state.status = "Pasted DBC for " + toString(sources);
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
  if (!session.settings_status().empty()) {
    ImGui::TextDisabled("%s", session.settings_status().c_str());
  }

  if (changed) save_state(&pane, state);

  draw_opendbc_browser(session, &state, &opendbc_cache, &changed);
  if (changed) save_state(&pane, state);

  ImGui::Separator();
  push_bold_font();
  ImGui::TextUnformatted("Loaded DBCs");
  pop_bold_font();
  draw_dbc_rows(session, &state, &assign_cache, &changed);
  if (changed) save_state(&pane, state);
}

}  // namespace loggy
