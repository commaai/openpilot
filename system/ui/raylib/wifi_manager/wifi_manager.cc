
#include "system/ui/raylib/wifi_manager/wifi_manager.h"
#define RAYGUI_IMPLEMENTATION
#define BLANK RAYLIB_BLANK
#include <cassert>

#include "system/ui/raylib/util.h"
#include "third_party/raylib/include/raygui.h"
WifiManager::WifiManager() {
  GuiSetFont(getFont());
  GuiSetStyle(DEFAULT, TEXT_SIZE, 40);
  GuiSetStyle(DEFAULT, BACKGROUND_COLOR, ColorToInt((Color){30, 30, 30, 255}));      // Dark grey background
  GuiSetStyle(DEFAULT, BASE_COLOR_NORMAL, ColorToInt((Color){50, 50, 50, 255}));     // Dark button background
  GuiSetStyle(DEFAULT, TEXT_COLOR_NORMAL, ColorToInt((Color){200, 200, 200, 255}));  // Light grey text

  loadWifiNetworksAsync();
}

void WifiManager::draw(const Rectangle& rect) {
  double current_time = GetTime();
  if (current_time - last_scan_time_ > 60.0) {  // 1 minute
    last_scan_time_ = current_time;
    loadWifiNetworksAsync();  // Rescan the Wi-Fi networks
  }

  std::unique_lock lock(mutex_);
  if (wifi_networks_.empty()) {
    GuiDrawText("Loading Wi-Fi networks...", rect, TEXT_ALIGN_CENTER, RAYLIB_WHITE);
    return;
  }

  if (is_connecting_) {
    showPasswordDialog();
    return;
  }

  // Handle scrollable panel for displaying Wi-Fi networks
  Rectangle content_rect = {0, 0, rect.width - 20, wifi_networks_.size() * item_height_};
  Rectangle scissor = {0};
  GuiScrollPanel(rect, nullptr, content_rect, &scroll_offset_, &scissor);

  // Draw Wi-Fi networks inside the scrollable area
  BeginScissorMode(scissor.x, scissor.y, scissor.width, scissor.height);
  for (int i = 0; i < wifi_networks_.size(); ++i) {
    float yPos = i * item_height_ + scroll_offset_.y;
    const auto& network = wifi_networks_[i];

    // Draw network SSID and buttons for each network
    GuiLabel((Rectangle){20, yPos, 500, item_height_}, network.ssid.c_str());
    if (network.connected) {
      GuiLabel((Rectangle){550, yPos + 3, 220, item_height_ - 6}, "Connected");
    } else if (GuiButton((Rectangle){550, yPos + 3, 180, item_height_ - 6}, "Connect")) {
      selected_network_index_ = i;
      memset(password_input_, 0, sizeof(password_input_));
      is_connecting_ = true;
    }
    if (saved_networks_.count(network.ssid) && GuiButton((Rectangle){780, yPos + 3, 150, item_height_ - 6}, "Forget")) {
      wifi::forget(network.ssid);
      saved_networks_.erase(network.ssid);
      wifi_networks_ = wifi::scan_networks();  // Refresh the list
    }
  }
  EndScissorMode();
}

void WifiManager::showPasswordDialog() {
  // TODO: Implement a keyboard input dialog
  const auto& ssid = wifi_networks_[selected_network_index_].ssid;
  int result = GuiTextInputBox(
      (Rectangle){GetScreenWidth() / 2.0f - 120, GetScreenHeight() / 2.0f - 60, 240, 140},
      ("Connect to " + ssid).c_str(), "Password:", "Ok;Cancel", password_input_, 128, NULL);
  is_connecting_ = (result < 0);
  if (result == 1) {
    wifi::connect(ssid, password_input_);
    saved_networks_.insert(ssid);
  }
}

void WifiManager::loadWifiNetworksAsync() {
  async_task_ = std::async(std::launch::async, [this]() {
    auto networks = wifi::scan_networks();
    auto known_networks = wifi::saved_networks();

    std::unique_lock lock(mutex_);
    wifi_networks_.swap(networks);
    saved_networks_.swap(known_networks);
  });
}
