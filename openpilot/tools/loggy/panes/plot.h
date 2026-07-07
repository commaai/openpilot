#pragma once

#include "tools/loggy/shell/pane.h"

#include <string>
#include <string_view>

namespace loggy {

struct Session;

void draw_plot_pane(Session &session, PaneInstance &pane);
void draw_plot_context_menu(Session &session, PaneInstance &pane);
// Append a series path to a plot pane's state_json (used by the shell's drag-to-plot targets).
std::string plot_state_with_added_series(std::string_view state_json, std::string_view path);

}  // namespace loggy
