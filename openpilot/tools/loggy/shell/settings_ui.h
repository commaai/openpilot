#pragma once

#include "tools/loggy/backend/session.h"
#include "tools/loggy/shell/settings.h"
#include "tools/loggy/shell/theme.h"

#include <array>
#include <string>

namespace loggy {

struct SettingsUiState {
  bool open_popup = false;
  std::array<char, kMaxSettingsValueBytes + 1> opendbc_root_buffer{};
  std::array<char, kMaxSettingsValueBytes + 1> dbc_override_buffer{};
  std::array<char, kMaxSettingsValueBytes + 1> map_cache_root_buffer{};
  int target_fps = kDefaultLoggyTargetFps;
  int theme_index = 0;
  bool show_frame_hud = true;
  bool natural_map_drag = true;
  std::string popup_status;
};

void sync_settings_popup_fields(const Session &session, int target_fps, SettingsUiState *state);
void request_settings_popup(const Session &session, int target_fps, SettingsUiState &state);
bool apply_settings_popup(Session &session, bool options_show_frame_hud, LoggyThemeKind &theme_kind,
                         int &target_fps, bool &show_frame_hud, SettingsUiState &state);
void draw_settings_popup(Session &session, bool close_requested, bool options_show_frame_hud, LoggyThemeKind &theme_kind,
                        int &target_fps, bool &show_frame_hud, SettingsUiState &state);

}  // namespace loggy

