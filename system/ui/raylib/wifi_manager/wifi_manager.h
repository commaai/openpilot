#pragma once

#include <atomic>
#include <future>
#include <mutex>
#include <optional>
#include <thread>

#include "system/ui/raylib/raylib.h"
#include "system/ui/raylib/wifi_manager/nmcli_backend.h"

class WifiManager {
public:
  WifiManager();
  void render(const Rectangle &rect);

protected:
  enum class ActionState {
    None,
    Connect,
    Connecting,
    Forget,
    Forgetting
  };

  bool connectToNetwork();
  void scanNetworksAsync();
  void rescanIfNeeded();
  bool forgetNetwork();
  void renderNetworkList(const Rectangle &rect);
  void renderNetworkItem(const Rectangle& rect, const Network &network);
  void connectToNetworkAsync(const std::string &ssid, const std::string &password = "");
  void initiateAction(const Network& network, ActionState action);

  std::mutex mutex_;
  std::atomic<ActionState> current_action_ = ActionState::None;
  std::future<void> async_scan_task_;
  std::future<void> async_connection_task_;
  std::future<void> async_forget_task_;
  std::vector<Network> available_networks_;
  std::set<std::string> saved_networks_;
  std::optional<Network> selected_network_;

  Vector2 scroll_offset_ = {0, 0};
  const float item_height_ = 160;
  char password_input_buffer_[128] = {};
  double last_scan_time_ = 0;
};
