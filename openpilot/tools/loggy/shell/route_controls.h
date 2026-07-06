#pragma once

#include "tools/loggy/backend/route.h"
#include "tools/loggy/backend/session.h"

#include <array>
#include <string>

namespace loggy {

struct RouteUiState {
  bool open_popup = false;
  std::array<char, 256> route_name_buffer{};
  std::array<char, 64> route_slice_buffer{};
  int route_selector_index = 0;
  std::string status;
};

RouteSelection current_route_selection(const Session &session);
void sync_route_popup_fields(const Session &session, RouteUiState &state);
void request_route_popup(const Session &session, RouteUiState &state);
bool restart_route_from_popup(Session &session, RouteUiState &state, std::string route_name, std::string status_prefix);
void draw_route_popup(Session &session, RouteUiState &state, bool close_requested);

}  // namespace loggy
