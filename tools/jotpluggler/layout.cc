#include "tools/jotpluggler/internal.h"
#include "system/hardware/hw.h"

#include <unistd.h>

namespace fs = std::filesystem;

namespace {

enum class ModalAction {
  None,
  Primary,
  Secondary,
};

struct FindSignalMatch {
  const std::string *path = nullptr;
  int score = 0;
};

struct DbcEditorSource {
  fs::path path;
  DbcEditorState::SourceKind kind = DbcEditorState::SourceKind::None;
};

StreamSourceConfig stream_source_config_from_ui(const UiState &state) {
  StreamSourceConfig source;
  source.kind = state.stream_source_kind;
  source.address = util::strip(state.stream_address_buffer);
  if (source.kind == StreamSourceKind::CerealLocal) {
    source.address = "127.0.0.1";
  } else {
    source.address = normalize_stream_address(std::move(source.address));
  }
  return source;
}

void open_queued_popup(bool &flag, const char *name) {
  if (flag) {
    ImGui::OpenPopup(name);
    flag = false;
  }
}

ModalAction draw_modal_action_row(const char *primary_label,
                                  const char *secondary_label = "Cancel",
                                  float width = 120.0f) {
  if (ImGui::Button(primary_label, ImVec2(width, 0.0f))) {
    return ModalAction::Primary;
  }
  ImGui::SameLine();
  if (ImGui::Button(secondary_label, ImVec2(width, 0.0f))) {
    return ModalAction::Secondary;
  }
  return ModalAction::None;
}

std::vector<FindSignalMatch> find_signal_matches(const AppSession &session, std::string_view query) {
  std::vector<FindSignalMatch> matches;
  if (query.empty()) {
    return matches;
  }
  const std::string needle = lowercase_copy(query);
  for (const std::string &path : session.route_data.paths) {
    const std::string hay = lowercase_copy(path);
    const size_t pos = hay.find(needle);
    if (pos == std::string::npos) {
      continue;
    }
    const size_t slash = path.find_last_of('/');
    const std::string_view label = slash == std::string::npos ? std::string_view(path) : std::string_view(path).substr(slash + 1);
    int score = static_cast<int>(pos * 8 + path.size());
    if (lowercase_copy(label) == needle) score -= 60;
    if (util::starts_with(hay, needle)) score -= 30;
    matches.push_back({.path = &path, .score = score});
  }
  std::sort(matches.begin(), matches.end(), [](const FindSignalMatch &a, const FindSignalMatch &b) {
    return std::tie(a.score, *a.path) < std::tie(b.score, *b.path);
  });
  if (matches.size() > 200) {
    matches.resize(200);
  }
  return matches;
}

bool open_find_signal_result(UiState *state, const std::string &path) {
  state->selected_browser_paths = {path};
  state->selected_browser_path = path;
  state->browser_selection_anchor = path;
  state->status_text = "Selected signal " + path;
  return true;
}

void draw_open_route_popup(AppSession *session, UiState *state) {
  if (!ImGui::BeginPopupModal("Open Route", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    return;
  }
  ImGui::TextUnformatted("Load a route into the current layout.");
  ImGui::Separator();
  input_text_string("Route", &state->route_buffer);
  input_text_string("Data Dir", &state->data_dir_buffer);
  ImGui::Spacing();
  switch (draw_modal_action_row("Load")) {
    case ModalAction::Primary:
      reload_session(session, state, state->route_buffer, state->data_dir_buffer);
      ImGui::CloseCurrentPopup();
      break;
    case ModalAction::Secondary:
      sync_route_buffers(state, *session);
      ImGui::CloseCurrentPopup();
      break;
    case ModalAction::None:
      break;
  }
  ImGui::EndPopup();
}

void draw_stream_popup(AppSession *session, UiState *state) {
  if (!ImGui::BeginPopupModal("Live Stream", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    return;
  }

  ImGui::TextUnformatted("Connect to a live source.");
  ImGui::Separator();
  if (ImGui::RadioButton("Local (MSGQ)", state->stream_source_kind == StreamSourceKind::CerealLocal)) {
    state->stream_source_kind = StreamSourceKind::CerealLocal;
  }
  if (ImGui::RadioButton("Remote (ZMQ)", state->stream_source_kind == StreamSourceKind::CerealRemote)) {
    state->stream_source_kind = StreamSourceKind::CerealRemote;
  }

  if (state->stream_source_kind == StreamSourceKind::CerealRemote) {
    input_text_string("Address", &state->stream_address_buffer);
  }
  ImGui::InputDouble("Buffer (seconds)", &state->stream_buffer_seconds, 0.0, 0.0, "%.0f");
  ImGui::Spacing();
  switch (draw_modal_action_row("Connect")) {
    case ModalAction::Primary: {
      const StreamSourceConfig source = stream_source_config_from_ui(*state);
      if (start_stream_session(session, state, source, state->stream_buffer_seconds, false)) {
        ImGui::CloseCurrentPopup();
      }
      break;
    }
    case ModalAction::Secondary:
      sync_stream_buffers(state, *session);
      ImGui::CloseCurrentPopup();
      break;
    case ModalAction::None:
      break;
  }
  ImGui::EndPopup();
}

void draw_load_layout_popup(AppSession *session, UiState *state) {
  if (!ImGui::BeginPopupModal("Load Layout", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    return;
  }
  ImGui::TextUnformatted("Load a JotPlugger JSON layout.");
  ImGui::Separator();
  input_text_string("Layout", &state->load_layout_buffer);
  ImGui::Spacing();
  switch (draw_modal_action_row("Load")) {
    case ModalAction::Primary:
      if (reload_layout(session, state, state->load_layout_buffer)) {
        ImGui::CloseCurrentPopup();
      }
      break;
    case ModalAction::Secondary:
      sync_layout_buffers(state, *session);
      ImGui::CloseCurrentPopup();
      break;
    case ModalAction::None:
      break;
  }
  ImGui::EndPopup();
}

void draw_save_layout_popup(AppSession *session, UiState *state) {
  if (!ImGui::BeginPopupModal("Save Layout", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    return;
  }
  ImGui::TextUnformatted("Save the current workspace as a JotPlugger JSON layout.");
  ImGui::Separator();
  input_text_string("Layout", &state->save_layout_buffer);
  ImGui::Spacing();
  switch (draw_modal_action_row("Save")) {
    case ModalAction::Primary:
      if (save_layout(session, state, state->save_layout_buffer)) {
        ImGui::CloseCurrentPopup();
      }
      break;
    case ModalAction::Secondary:
      sync_layout_buffers(state, *session);
      ImGui::CloseCurrentPopup();
      break;
    case ModalAction::None:
      break;
  }
  ImGui::EndPopup();
}

void draw_preferences_popup(AppSession *session, UiState *state) {
  if (!ImGui::BeginPopupModal("Preferences", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    return;
  }
  if (session->map_data) {
    const MapCacheStats map_cache = session->map_data->cacheStats();
    const MapCacheStats download_cache = directory_cache_stats(Path::download_cache_root());
    ImGui::TextUnformatted("Map");
    ImGui::Separator();
    ImGui::Text("Map cache: %s in %zu file%s",
                format_cache_bytes(map_cache.bytes).c_str(),
                map_cache.files,
                map_cache.files == 1 ? "" : "s");
    if (ImGui::Button("Clear Map Cache", ImVec2(120.0f, 0.0f))) {
      session->map_data->clearCache();
      state->status_text = "Cleared map cache";
    }
    ImGui::Spacing();
    ImGui::TextUnformatted("comma Download Cache");
    ImGui::Separator();
    ImGui::Text("Download cache: %s in %zu file%s",
                format_cache_bytes(download_cache.bytes).c_str(),
                download_cache.files,
                download_cache.files == 1 ? "" : "s");
    ImGui::TextDisabled("%s", Path::download_cache_root().c_str());
    ImGui::Spacing();
  }
  if (ImGui::Button("Close", ImVec2(120.0f, 0.0f))) {
    ImGui::CloseCurrentPopup();
  }
  ImGui::EndPopup();
}

void draw_find_signal_popup(AppSession *session, UiState *state) {
  if (!ImGui::BeginPopupModal("Find Signal", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    return;
  }
  ImGui::TextUnformatted("Search decoded signals across the loaded route.");
  ImGui::Separator();
  ImGui::SetNextItemWidth(560.0f);
  input_text_with_hint_string("##find_signal_query", "Search signal path or name...", &state->find_signal_buffer);
  if (ImGui::IsWindowAppearing()) {
    ImGui::SetKeyboardFocusHere(-1);
  }
  const std::vector<FindSignalMatch> matches = find_signal_matches(*session, state->find_signal_buffer);
  ImGui::Spacing();
  ImGui::TextDisabled("%zu match%s", matches.size(), matches.size() == 1 ? "" : "es");
  if (ImGui::BeginChild("##find_signal_results", ImVec2(760.0f, 360.0f), true)) {
    for (const FindSignalMatch &match : matches) {
      const std::string &path = *match.path;
      const size_t slash = path.find_last_of('/');
      const std::string_view label = slash == std::string::npos ? std::string_view(path) : std::string_view(path).substr(slash + 1);
      if (ImGui::Selectable((std::string(label) + "##" + path).c_str(), false, ImGuiSelectableFlags_SpanAllColumns)) {
        if (open_find_signal_result(state, path)) {
          ImGui::CloseCurrentPopup();
        }
      }
      ImGui::SameLine(280.0f);
      ImGui::TextDisabled("%s", path.c_str());
    }
  }
  ImGui::EndChild();
  ImGui::Spacing();
  if (ImGui::Button("Close", ImVec2(120.0f, 0.0f))) {
    ImGui::CloseCurrentPopup();
  }
  ImGui::EndPopup();
}

std::string default_dbc_template() {
  return "VERSION \"\"\n\nNS_ :\nBS_:\nBU_: XXX\n";
}

DbcEditorSource resolve_dbc_editor_source(const std::string &dbc_name) {
  const fs::path generated_dbc_dir = repo_root() / "tools" / "jotpluggler" / "generated_dbcs";
  const std::array<DbcEditorSource, 2> candidates = {{
    {.path = repo_root() / "opendbc" / "dbc" / (dbc_name + ".dbc"), .kind = DbcEditorState::SourceKind::Opendbc},
    {.path = generated_dbc_dir / (dbc_name + ".dbc"), .kind = DbcEditorState::SourceKind::Generated},
  }};
  for (const DbcEditorSource &candidate : candidates) {
    if (fs::exists(candidate.path)) {
      return candidate;
    }
  }
  return {};
}

void load_dbc_editor_state(const AppSession &session, UiState *state) {
  DbcEditorState &editor = state->dbc_editor;
  const std::string dbc_name = !session.dbc_override.empty() ? session.dbc_override : session.route_data.dbc_name;
  editor.source_name = dbc_name.empty() ? "untitled" : dbc_name;
  editor.source_path.clear();
  editor.source_kind = DbcEditorState::SourceKind::None;
  if (dbc_name.empty()) {
    editor.save_name = "custom_can";
    editor.text = default_dbc_template();
  } else {
    const DbcEditorSource source = resolve_dbc_editor_source(dbc_name);
    editor.source_kind = source.kind;
    editor.source_path = source.path;
    editor.text = source.path.empty() ? default_dbc_template() : read_file_or_throw(source.path);
    editor.save_name = source.kind == DbcEditorState::SourceKind::Generated ? dbc_name : dbc_name + "_edited";
  }
  editor.loaded = true;
}

bool ensure_dbc_editor_loaded(const AppSession &session, UiState *state) {
  if (!state->dbc_editor.loaded) {
    try {
      load_dbc_editor_state(session, state);
    } catch (const std::exception &err) {
      state->error_text = err.what();
      state->open_error_popup = true;
      return false;
    }
  }
  return true;
}

bool save_dbc_editor_contents(AppSession *session, UiState *state) {
  DbcEditorState &editor = state->dbc_editor;
  editor.save_name = util::strip(editor.save_name);
  if (editor.save_name.empty()) {
    state->error_text = "DBC name cannot be empty";
    state->open_error_popup = true;
    return false;
  }
  if (editor.source_kind == DbcEditorState::SourceKind::Opendbc && editor.save_name == editor.source_name) {
    state->error_text = "Save edited opendbc files under a new name";
    state->open_error_popup = true;
    return false;
  }
  try {
    dbc::Database::fromContent(editor.text, editor.save_name + ".dbc");
    const fs::path generated_dbc_dir = repo_root() / "tools" / "jotpluggler" / "generated_dbcs";
    fs::create_directories(generated_dbc_dir);
    const fs::path output = generated_dbc_dir / (editor.save_name + ".dbc");
    write_file_or_throw(output, editor.text);
    apply_dbc_override_change(session, state, editor.save_name);
    editor.source_name = editor.save_name;
    editor.source_path = output;
    editor.source_kind = DbcEditorState::SourceKind::Generated;
    editor.loaded = false;
    state->status_text = "Saved DBC " + editor.save_name;
    return true;
  } catch (const std::exception &err) {
    state->error_text = err.what();
    state->open_error_popup = true;
    return false;
  }
}

void draw_dbc_editor_popup(AppSession *session, UiState *state) {
  if (!ImGui::BeginPopupModal("DBC Editor", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    return;
  }
  DbcEditorState &editor = state->dbc_editor;
  if (!ensure_dbc_editor_loaded(*session, state)) {
    ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
    return;
  }
  ImGui::TextUnformatted("Edit DBC text and save it into generated_dbcs.");
  ImGui::Separator();
  ImGui::SetNextItemWidth(260.0f);
  input_text_string("DBC Name", &editor.save_name, ImGuiInputTextFlags_AutoSelectAll);
  if (!editor.source_path.empty()) {
    ImGui::TextDisabled("%s", editor.source_path.string().c_str());
  } else {
    ImGui::TextDisabled("New in-memory DBC");
  }
  ImGui::Spacing();
  input_text_multiline_string("##dbc_editor_text", &editor.text, ImVec2(920.0f, 520.0f), ImGuiInputTextFlags_AllowTabInput);
  ImGui::Spacing();
  if (ImGui::Button("Apply + Save", ImVec2(140.0f, 0.0f))) {
    if (save_dbc_editor_contents(session, state)) {
      ImGui::CloseCurrentPopup();
    }
  }
  ImGui::SameLine();
  if (ImGui::Button("Reload Source", ImVec2(140.0f, 0.0f))) {
    editor.loaded = false;
  }
  ImGui::SameLine();
  if (ImGui::Button("Close", ImVec2(120.0f, 0.0f))) {
    editor.loaded = false;
    ImGui::CloseCurrentPopup();
  }
  ImGui::EndPopup();
}

void draw_axis_limits_popup(AppSession *session, UiState *state) {
  if (!ImGui::BeginPopupModal("Edit Axis Limits", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    return;
  }
  const WorkspaceTab *tab = app_active_tab(session->layout, *state);
  const bool valid_pane = tab != nullptr
    && state->axis_limits.pane_index >= 0
    && state->axis_limits.pane_index < static_cast<int>(tab->panes.size());
  if (!valid_pane) {
    ImGui::TextWrapped("The selected pane is no longer available.");
    ImGui::Spacing();
    if (ImGui::Button("Close", ImVec2(120.0f, 0.0f))) {
      state->axis_limits.pane_index = -1;
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
    return;
  }

  ImGui::TextUnformatted("X range applies to the active tab. Y limits apply to the selected pane.");
  ImGui::Separator();
  ImGui::TextUnformatted("Horizontal");
  ImGui::SetNextItemWidth(180.0f);
  ImGui::InputDouble("X Min", &state->axis_limits.x_min, 0.0, 0.0, "%.3f");
  ImGui::SetNextItemWidth(180.0f);
  ImGui::InputDouble("X Max", &state->axis_limits.x_max, 0.0, 0.0, "%.3f");
  ImGui::Spacing();
  ImGui::TextUnformatted("Vertical");
  ImGui::Checkbox("Use Y Min", &state->axis_limits.y_min_enabled);
  ImGui::BeginDisabled(!state->axis_limits.y_min_enabled);
  ImGui::SetNextItemWidth(180.0f);
  ImGui::InputDouble("Y Min", &state->axis_limits.y_min, 0.0, 0.0, "%.6g");
  ImGui::EndDisabled();
  ImGui::Checkbox("Use Y Max", &state->axis_limits.y_max_enabled);
  ImGui::BeginDisabled(!state->axis_limits.y_max_enabled);
  ImGui::SetNextItemWidth(180.0f);
  ImGui::InputDouble("Y Max", &state->axis_limits.y_max, 0.0, 0.0, "%.6g");
  ImGui::EndDisabled();
  ImGui::Spacing();
  switch (draw_modal_action_row("Apply")) {
    case ModalAction::Primary:
      if (apply_axis_limits_editor(session, state)) {
        state->axis_limits.pane_index = -1;
        ImGui::CloseCurrentPopup();
      }
      break;
    case ModalAction::Secondary:
      state->axis_limits.pane_index = -1;
      ImGui::CloseCurrentPopup();
      break;
    case ModalAction::None:
      break;
  }
  ImGui::EndPopup();
}

void draw_error_popup(UiState *state) {
  if (state->open_error_popup) {
    ImGui::OpenPopup("Error");
    state->open_error_popup = false;
  }
  if (!ImGui::BeginPopupModal("Error", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    return;
  }
  ImGui::TextWrapped("%s", state->error_text.c_str());
  ImGui::Spacing();
  if (ImGui::Button("Close", ImVec2(120.0f, 0.0f))) {
    ImGui::CloseCurrentPopup();
  }
  ImGui::EndPopup();
}

}  // namespace

bool reset_layout(AppSession *session, UiState *state) {
  try {
    if (session->layout_path.empty()) {
      start_new_layout(session, state, "Reset layout");
      return true;
    }
    clear_layout_autosave(*session);
    session->layout = load_sketch_layout(session->layout_path);
    state->layout_dirty = false;
    session->autosave_path = autosave_path_for_layout(session->layout_path);
    state->undo.reset(session->layout);
    refresh_replaced_layout_ui(session, state, false);
    reset_shared_range(state, *session);
    state->status_text = "Reset layout";
    return true;
  } catch (const std::exception &err) {
    state->error_text = err.what();
    state->open_error_popup = true;
    state->status_text = "Failed to reset layout";
    return false;
  }
}

bool reload_layout(AppSession *session, UiState *state, const std::string &layout_arg) {
  try {
    const bool preserve_shared_range = session->route_data.has_time_range && state->has_shared_range;
    const double preserved_x_min = state->x_view_min;
    const double preserved_x_max = state->x_view_max;
    const fs::path layout_path = resolve_layout_path(layout_arg);
    session->autosave_path = autosave_path_for_layout(layout_path);
    const bool load_draft = fs::exists(session->autosave_path);
    session->layout = load_sketch_layout(load_draft ? session->autosave_path : layout_path);
    session->layout_path = layout_path;
    state->layout_dirty = load_draft;
    state->undo.reset(session->layout);
    refresh_replaced_layout_ui(session, state, true);
    if (preserve_shared_range) {
      state->has_shared_range = true;
      state->x_view_min = preserved_x_min;
      state->x_view_max = preserved_x_max;
      clamp_shared_range(state, *session);
    } else {
      reset_shared_range(state, *session);
    }
    state->status_text = std::string(load_draft ? "Loaded layout draft " : "Loaded layout ")
      + layout_path.filename().string();
    return true;
  } catch (const std::exception &err) {
    state->error_text = err.what();
    state->open_error_popup = true;
    state->status_text = "Failed to load layout";
    return false;
  }
}

bool save_layout(AppSession *session, UiState *state, const std::string &layout_path) {
  try {
    if (layout_path.empty()) throw std::runtime_error("Layout path is empty");
    session->layout.current_tab_index = state->active_tab_index;
    const fs::path previous_autosave = session->autosave_path;
    const fs::path output = fs::absolute(fs::path(layout_path));
    save_layout_json(session->layout, output);
    session->layout_path = output;
    session->autosave_path = autosave_path_for_layout(output);
    if (!previous_autosave.empty() && previous_autosave != session->autosave_path && fs::exists(previous_autosave)) {
      fs::remove(previous_autosave);
    }
    clear_layout_autosave(*session);
    state->layout_dirty = false;
    sync_layout_buffers(state, *session);
    state->status_text = "Saved layout " + output.filename().string();
    return true;
  } catch (const std::exception &err) {
    state->error_text = err.what();
    state->open_error_popup = true;
    state->status_text = "Failed to save layout";
    return false;
  }
}

void rebuild_session_route_data(AppSession *session, UiState *state,
                                const RouteLoadProgressCallback &progress) {
  apply_route_data(session, state, load_route_data(session->route_name, session->data_dir, session->dbc_override, progress));
}

void stop_stream_session(AppSession *session, UiState *state, bool preserve_data) {
  if (preserve_data && session->stream_poller && session->data_mode == SessionDataMode::Stream) {
    session->stream_poller->setPaused(true);
  } else if (session->stream_poller) {
    session->stream_poller->stop();
  }
  session->stream_paused = preserve_data && session->data_mode == SessionDataMode::Stream;
  if (!preserve_data) {
    session->stream_time_offset.reset();
    apply_route_data(session, state, RouteData{});
  }
  sync_stream_buffers(state, *session);
}

bool start_stream_session(AppSession *session,
                          UiState *state,
                          const StreamSourceConfig &source,
                          double buffer_seconds,
                          bool preserve_existing_data) {
  try {
    if (session->route_loader) {
      session->route_loader.reset();
    }
    session->data_mode = SessionDataMode::Stream;
    session->route_id = {};
    session->route_name.clear();
    session->data_dir.clear();
    session->stream_source = source;
    if (session->stream_source.kind == StreamSourceKind::CerealLocal) {
      session->stream_source.address = "127.0.0.1";
    }
    session->stream_buffer_seconds = std::max(1.0, buffer_seconds);
    session->next_stream_custom_refresh_time = 0.0;
    session->stream_paused = false;
    if (preserve_existing_data && session->stream_poller) {
      StreamPollSnapshot snapshot = session->stream_poller->snapshot();
      if (snapshot.active) {
        session->stream_poller->setPaused(false);
        sync_route_buffers(state, *session);
        sync_stream_buffers(state, *session);
        state->follow_latest = true;
        state->playback_playing = false;
        state->status_text = "Resumed stream " + stream_source_target_label(session->stream_source);
        return true;
      }
    }
    if (!preserve_existing_data) {
      session->stream_time_offset.reset();
      apply_route_data(session, state, RouteData{});
    }
    if (!session->stream_poller) {
      session->stream_poller = std::make_unique<StreamPoller>();
    }
    session->stream_poller->start(session->stream_source,
                                  session->stream_buffer_seconds,
                                  session->dbc_override,
                                  session->stream_time_offset);
    sync_route_buffers(state, *session);
    sync_stream_buffers(state, *session);
    state->follow_latest = true;
    state->playback_playing = false;
    state->status_text = preserve_existing_data ? "Resumed stream " + stream_source_target_label(session->stream_source)
                                                : "Streaming from " + stream_source_target_label(session->stream_source);
    return true;
  } catch (const std::exception &err) {
    state->error_text = err.what();
    state->open_error_popup = true;
    state->status_text = "Failed to start stream";
    return false;
  }
}

void start_async_route_load(AppSession *session, UiState *state) {
  if (!session->route_loader) {
    return;
  }
  apply_route_data(session, state, RouteData{});
  session->route_loader->start(session->route_name, session->data_dir, session->dbc_override);
  state->status_text = session->route_name.empty() ? "Ready" : "Loading route " + session->route_name;
}

void poll_async_route_load(AppSession *session, UiState *state) {
  if (!session->route_loader) {
    return;
  }
  RouteData loaded_route;
  std::string error_text;
  if (!session->route_loader->consume(&loaded_route, &error_text)) {
    return;
  }
  if (!error_text.empty()) {
    state->error_text = error_text;
    state->open_error_popup = true;
    state->status_text = "Failed to load route";
    return;
  }
  apply_route_data(session, state, std::move(loaded_route));
  state->status_text = session->route_name.empty() ? "Ready" : "Loaded route " + session->route_name;
}

bool reload_session(AppSession *session, UiState *state, const std::string &route_name, const std::string &data_dir) {
  try {
    stop_stream_session(session, state, false);
    session->data_mode = SessionDataMode::Route;
    session->route_name = route_name;
    session->route_id = parse_route_identifier(route_name);
    session->data_dir = data_dir;
    if (session->async_route_loading) {
      if (!session->route_loader) {
        session->route_loader = std::make_unique<AsyncRouteLoader>(::isatty(STDERR_FILENO) != 0);
      }
      start_async_route_load(session, state);
    } else {
      rebuild_session_route_data(session, state);
      state->status_text = "Loaded route " + route_name;
    }
    sync_route_buffers(state, *session);
    return true;
  } catch (const std::exception &err) {
    state->error_text = err.what();
    state->open_error_popup = true;
    state->status_text = "Failed to load route";
    return false;
  }
}

void draw_popups(AppSession *session, UiState *state) {
  open_queued_popup(state->open_open_route, "Open Route");
  if (state->open_stream) {
    sync_stream_buffers(state, *session);
  }
  open_queued_popup(state->open_stream, "Live Stream");
  if (state->open_load_layout || state->open_save_layout) {
    sync_layout_buffers(state, *session);
  }
  open_queued_popup(state->open_load_layout, "Load Layout");
  open_queued_popup(state->open_save_layout, "Save Layout");
  open_queued_popup(state->open_preferences, "Preferences");
  open_queued_popup(state->dbc_editor.open, "DBC Editor");
  open_queued_popup(state->open_find_signal, "Find Signal");
  open_queued_popup(state->axis_limits.open, "Edit Axis Limits");

  draw_open_route_popup(session, state);
  draw_stream_popup(session, state);
  draw_load_layout_popup(session, state);
  draw_save_layout_popup(session, state);
  draw_preferences_popup(session, state);
  draw_dbc_editor_popup(session, state);
  draw_find_signal_popup(session, state);
  draw_axis_limits_popup(session, state);
  draw_error_popup(state);
}
