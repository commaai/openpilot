#pragma once

#include "tools/loggy/shell/pane.h"

#include <filesystem>
#include <string_view>

namespace loggy {

std::filesystem::path map_basemap_effective_cache_root(std::string_view configured_root);
void draw_map_pane(Session &session, PaneInstance &pane);

}  // namespace loggy
