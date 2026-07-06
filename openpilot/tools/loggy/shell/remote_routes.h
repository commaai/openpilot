#pragma once

#include <string_view>

namespace loggy {

struct RemoteRouteBrowserActions {
  void (*open_route)(void *ctx, std::string_view route) = nullptr;
  void *ctx = nullptr;
};

void open_remote_route_browser();
void draw_remote_route_browser(const RemoteRouteBrowserActions &actions);

}  // namespace loggy
