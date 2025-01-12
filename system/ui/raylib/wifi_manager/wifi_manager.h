#pragma once

#include <string>
#include <vector>

#include "system/ui/raylib/raylib.h"
#include "system/ui/raylib/wifi_manager/nmcli_backend.h"

class WifiManager {
public:
  WifiManager();
  void draw(const Rectangle &rect);

protected:
  void showPasswordDialog();

  std::vector<Network> wifi_networks_;
  Vector2 scroll_offset_ = {0, 0};
  int selected_network_index_ = 0;
  const float item_height_ = 40;
  bool is_connecting_ = false;
  char password_input_[128] = {};
};
