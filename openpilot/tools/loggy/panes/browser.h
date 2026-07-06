#pragma once

#include "tools/loggy/shell/pane.h"

namespace loggy {

inline constexpr const char *kLoggySeriesPathPayload = "LOGGY_SERIES_PATH";

void draw_browser_pane(Session &session, PaneInstance &pane);

}  // namespace loggy
