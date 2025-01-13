#pragma once

#include <future>
#include <mutex>
#include <thread>

#include "system/ui/raylib/raylib.h"
#include "system/ui/raylib/wifi_manager/nmcli_backend.h"

class WifiManager {
public:
  WifiManager();
  void draw(const Rectangle &rect);

protected:
  void showPasswordDialog();
  void loadWifiNetworksAsync();

  std::mutex mutex_;
  std::future<void> async_task_;
  std::vector<Network> wifi_networks_;
  std::set<std::string> saved_networks_;
  double last_scan_time_ = 0;

  Vector2 scroll_offset_ = {0, 0};
  int selected_network_index_ = 0;
  const float item_height_ = 60;
  bool is_connecting_ = false;
  char password_input_[128] = {};

};
