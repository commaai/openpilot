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
  void scanNetworksAsync();
  void rescanIfNeeded();
  void drawNetworkList(const Rectangle &rect);
  void drawNetworkItem(const Rectangle& rect, const Network &network);
  void initiateConnection(const std::string &ssid);
  void forgetNetwork(const std::string& ssid);

  std::mutex mutex_;
  std::future<void> async_task_;
  std::vector<Network> wifi_networks_;
  std::set<std::string> saved_networks_;
  double last_scan_time_ = 0;

  Vector2 scroll_offset_ = {0, 0};
  std::string connecting_ssid_;
  const float item_height_ = 160;
  bool requires_password_ = false;
  char password_input_[128] = {};
};
