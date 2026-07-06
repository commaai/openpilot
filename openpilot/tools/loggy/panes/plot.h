#pragma once

#include "tools/loggy/shell/pane.h"

namespace loggy {

struct Session;

void draw_plot_pane(Session &session, PaneInstance &pane);

}  // namespace loggy
