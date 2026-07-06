#include "tools/loggy/shell/settings_ui.h"

#include "tools/loggy/panes/map.h"
#include "tools/loggy/shell/theme.h"

#include "imgui.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <filesystem>
#include <string>

namespace loggy {
namespace {

namespace fs = std::filesystem;

constexpr std::array<ThemeKind, 2> kThemeOptions = {
  ThemeKind::Darcula,
  ThemeKind::Light,
};

int theme_kind_to_index(ThemeKind theme) {
  for (size_t i = 0; i < kThemeOptions.size(); ++i) {
    if (kThemeOptions[i] == theme) return static_cast<int>(i);
  }
  return 0;
}

}  // namespace

void sync_settings_popup_fields(const Session &session, int target_fps, SettingsUiState *state) {
  if (state == nullptr) return;
  const LoggySettings &settings = session.settings;
  state->opendbc_root_text = settings.opendbc_root;
  state->dbc_override_text = settings.dbc_override;
  state->map_cache_root_text = settings.map_cache_root;
  state->target_fps = std::clamp(target_fps, kMinLoggyTargetFps, kMaxLoggyTargetFps);
  state->theme_index = theme_kind_to_index(theme_from_name(settings.theme));
  state->show_frame_hud = settings.show_frame_hud;
  state->natural_map_drag = settings.natural_map_drag;
  state->popup_status = session.settings_status;
}

void request_settings_popup(const Session &session, int target_fps, SettingsUiState &state) {
  sync_settings_popup_fields(session, target_fps, &state);
  state.open_popup = true;
}

bool apply_settings_popup(Session &session, bool options_show_frame_hud, ThemeKind &theme_kind,
                         int &target_fps, bool &show_frame_hud, SettingsUiState &state) {
  LoggySettings &settings = session.settings;
  const LoggySettings previous_settings = settings;

  const std::string opendbc_root = state.opendbc_root_text;
  const std::string dbc_override = state.dbc_override_text;
  const std::string map_cache_root = state.map_cache_root_text;
  state.theme_index = std::clamp(state.theme_index, 0, static_cast<int>(kThemeOptions.size()) - 1);
  const ThemeKind next_theme = kThemeOptions[static_cast<size_t>(state.theme_index)];
  const std::string next_theme_name = theme_name(next_theme);
  const int next_target_fps = std::clamp(state.target_fps, kMinLoggyTargetFps, kMaxLoggyTargetFps);
  const bool next_show_frame_hud = state.show_frame_hud;
  const bool next_natural_map_drag = state.natural_map_drag;

  const bool root_changed = settings.opendbc_root != opendbc_root;
  const bool override_changed = settings.dbc_override != dbc_override;
  const bool app_changed = settings.target_fps != next_target_fps || settings.show_frame_hud != next_show_frame_hud ||
                           settings.map_cache_root != map_cache_root || settings.natural_map_drag != next_natural_map_drag ||
                           settings.theme != next_theme_name;

  settings.opendbc_root = opendbc_root;
  settings.map_cache_root = map_cache_root;
  settings.theme = next_theme_name;
  settings.target_fps = next_target_fps;
  settings.show_frame_hud = next_show_frame_hud;
  settings.natural_map_drag = next_natural_map_drag;
  normalize_loggy_settings(&settings);

  if (override_changed) {
    std::string error;
    if (!session.set_manual_dbc_name(dbc_override, error)) {
      const std::string message = error.empty() ? "DBC override rejected" : error;
      settings = previous_settings;
      sync_settings_popup_fields(session, target_fps, &state);
      state.popup_status = message;
      return false;
    }
    target_fps = settings.target_fps;
    theme_kind = theme_from_name(settings.theme);
    apply_theme(theme_kind);
    show_frame_hud = options_show_frame_hud && settings.show_frame_hud;
    state.popup_status = session.settings_status;
    return true;
  }

  if (root_changed || app_changed) {
    std::string error;
    if (!session.save_settings(error)) {
      settings = previous_settings;
      state.popup_status = error.empty() ? "Settings save failed" : error;
      return false;
    }
  }
  target_fps = settings.target_fps;
  theme_kind = theme_from_name(settings.theme);
  apply_theme(theme_kind);
  show_frame_hud = options_show_frame_hud && settings.show_frame_hud;
  state.popup_status = session.settings_status;
  return true;
}

void draw_settings_popup(Session &session, bool close_requested, bool options_show_frame_hud, ThemeKind &theme_kind,
                        int &target_fps, bool &show_frame_hud, SettingsUiState &state) {
  if (state.open_popup) {
    ImGui::OpenPopup("Settings");
    state.open_popup = false;
  }

  ImGui::SetNextWindowSize(ImVec2(560.0f, 0.0f), ImGuiCond_Appearing);
  if (!ImGui::BeginPopupModal("Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) return;

  ImGui::TextDisabled("Config: %s", session.settings_path.string().c_str());
  if (!session.active_dbc_name.empty() || !session.auto_dbc_name.empty()) {
    const std::string active = session.active_dbc_name.empty() ? "--" : session.active_dbc_name;
    ImGui::TextDisabled("Active DBC: %s", active.c_str());
  }
  if (!session.dbc_status.empty()) ImGui::TextDisabled("%s", session.dbc_status.c_str());
  ImGui::Separator();

  ImGui::TextUnformatted("opendbc root");
  ImGui::SetNextItemWidth(-1.0f);
  input_text_with_hint("##settings_opendbc_root", "opendbc_repo/opendbc/dbc", &state.opendbc_root_text);
  ImGui::TextUnformatted("DBC override");
  ImGui::SetNextItemWidth(-1.0f);
  input_text_with_hint("##settings_dbc_override", "dbc name or /path/to.dbc", &state.dbc_override_text);
  ImGui::Spacing();
  ImGui::SeparatorText("App");
  state.theme_index = std::clamp(state.theme_index, 0, static_cast<int>(kThemeOptions.size()) - 1);
  const ThemeKind selected_theme = kThemeOptions[static_cast<size_t>(state.theme_index)];
  ImGui::SetNextItemWidth(180.0f);
  if (ImGui::BeginCombo("Theme", theme_label(selected_theme))) {
    for (int i = 0; i < static_cast<int>(kThemeOptions.size()); ++i) {
      const ThemeKind option = kThemeOptions[static_cast<size_t>(i)];
      const bool selected = i == state.theme_index;
      if (ImGui::Selectable(theme_label(option), selected)) state.theme_index = i;
      if (selected) ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }
  ImGui::SetNextItemWidth(120.0f);
  ImGui::InputInt("Target FPS", &state.target_fps, 5, 15);
  state.target_fps = std::clamp(state.target_fps, kMinLoggyTargetFps, kMaxLoggyTargetFps);
  ImGui::SameLine();
  ImGui::TextDisabled("%d-%d", kMinLoggyTargetFps, kMaxLoggyTargetFps);
  ImGui::Checkbox("Frame-Time HUD", &state.show_frame_hud);
  ImGui::Checkbox("Natural map drag", &state.natural_map_drag);
  ImGui::TextUnformatted("Map cache root");
  ImGui::SetNextItemWidth(-110.0f);
  input_text_with_hint("##settings_map_cache_root", "(default)", &state.map_cache_root_text);
  ImGui::SameLine();
  if (ImGui::Button("Default", ImVec2(96.0f, 0.0f))) {
    state.map_cache_root_text.clear();
  }
  const fs::path effective_cache_root = map_basemap_effective_cache_root(state.map_cache_root_text);
  ImGui::TextDisabled("Effective: %s", effective_cache_root.string().c_str());

  ImGui::Spacing();
  if (ImGui::Button("Save", ImVec2(100.0f, 0.0f))) {
    if (apply_settings_popup(session, options_show_frame_hud, theme_kind, target_fps, show_frame_hud, state)) ImGui::CloseCurrentPopup();
  }
  ImGui::SameLine();
  if (ImGui::Button("Reload", ImVec2(100.0f, 0.0f))) {
    sync_settings_popup_fields(session, target_fps, &state);
  }
  ImGui::SameLine();
  if (ImGui::Button("Clear Override", ImVec2(124.0f, 0.0f))) {
    state.dbc_override_text.clear();
    apply_settings_popup(session, options_show_frame_hud, theme_kind, target_fps, show_frame_hud, state);
  }
  ImGui::SameLine();
  if (ImGui::Button("Cancel", ImVec2(100.0f, 0.0f)) || close_requested) {
    ImGui::CloseCurrentPopup();
  }

  if (!state.popup_status.empty()) {
    ImGui::TextDisabled("%s", state.popup_status.c_str());
  }

  ImGui::EndPopup();
}

}  // namespace loggy
