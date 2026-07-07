#pragma once

#include "tools/loggy/shell/pane.h"

#include <string>
#include <string_view>

namespace loggy {

inline constexpr const char *kLoggySeriesPathPayload = "LOGGY_SERIES_PATH";
// Dragging a special source (Map / a camera) from the browser: payload carries a small id like
// "map" or "camera_road"; a pane accepting it converts itself to that pane type.
inline constexpr const char *kLoggySpecialItemPayload = "LOGGY_SPECIAL_ITEM";

struct SpecialItemPane {
  std::string type;
  std::string title;
  std::string state_json;
};

void draw_browser_pane(Session &session, PaneInstance &pane);

// Resolve a special-item drag payload id (e.g. "map", "camera_road") to the pane it should
// become; empty type means the id was not a special item. Used by the shell's pane drop target.
SpecialItemPane browser_special_item_pane(std::string_view id);

}  // namespace loggy
