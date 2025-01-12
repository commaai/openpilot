
#include "system/ui/raylib/wifi_manager/wifi_manager.h"
#define RAYGUI_IMPLEMENTATION
#include "raygui.h"  // Raylib's GUI extension
#include "system/ui/raylib/util.h"

WifiManager::WifiManager() {
  auto font = getFont();
  font.baseSize = 50;
  GuiSetFont(font);
  wifi_networks_ = list_wifi_networks();
}

void WifiManager::draw(const Rectangle& rect) {
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
    // Draw network SSID and buttons for each network
    float yPos = i * item_height_ + scroll_offset_.y;
    GuiLabel((Rectangle){20, yPos, 500, item_height_}, wifi_networks_[i].ssid.c_str());
    if (GuiButton((Rectangle){550, yPos + 2, 80, item_height_ - 4}, "Connect")) {
      selected_network_index_ = i;
      memset(password_input_, 0, sizeof(password_input_));
      is_connecting_ = true;
    }
    if (GuiButton((Rectangle){670, yPos + 2, 80, item_height_ - 4}, "Forget")) {
      forget_wifi(wifi_networks_[i].ssid);
      wifi_networks_ = list_wifi_networks();  // Refresh the list
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
    connect_to_wifi(ssid, password_input_);
  }
}
