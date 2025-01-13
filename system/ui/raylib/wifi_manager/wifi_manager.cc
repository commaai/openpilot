#include "system/ui/raylib/wifi_manager/wifi_manager.h"

#include "system/ui/raylib/util.h"
#define RAYGUI_IMPLEMENTATION
#define BLANK RAYLIB_BLANK
#include "third_party/raylib/include/raygui.h"

WifiManager::WifiManager() {
  GuiSetFont(getFont());
  GuiSetStyle(DEFAULT, TEXT_SIZE, 40);
  GuiSetStyle(DEFAULT, BACKGROUND_COLOR, ColorToInt({30, 30, 30, 255}));
  GuiSetStyle(DEFAULT, BASE_COLOR_NORMAL, ColorToInt({50, 50, 50, 255}));
  GuiSetStyle(DEFAULT, TEXT_COLOR_NORMAL, ColorToInt({200, 200, 200, 255}));

  scanNetworksAsync();
}

void WifiManager::draw(const Rectangle& rect) {
  rescanIfNeeded();

  std::unique_lock lock(mutex_);
  if (wifi_networks_.empty()) {
    GuiDrawText("Loading Wi-Fi networks...", rect, TEXT_ALIGN_CENTER, RAYLIB_WHITE);
    return;
  }

  if (requires_password_) {
    showPasswordDialog();
  } else {
    drawNetworkList(rect);
  }
}

void WifiManager::rescanIfNeeded() {
  double current_time = GetTime();
  if (current_time - last_scan_time_ > 60.0) {  // Rescan after 1 minute
    last_scan_time_ = current_time;
    scanNetworksAsync();
  }
}

void WifiManager::drawNetworkList(const Rectangle& rect) {
  Rectangle content_rect = {rect.x, rect.y, rect.width - 20, wifi_networks_.size() * item_height_};
  Rectangle scissor = {0};
  GuiScrollPanel(rect, nullptr, content_rect, &scroll_offset_, &scissor);

  const int padding = 20;
  BeginScissorMode(scissor.x, scissor.y, scissor.width, scissor.height);

  for (size_t i = 0; i < wifi_networks_.size(); ++i) {
    float y = content_rect.y + i * item_height_ + scroll_offset_.y;
    Rectangle item_rect = {content_rect.x + padding, y, content_rect.width - padding * 2, item_height_};

    drawNetworkItem(item_rect, wifi_networks_[i]);

    if (i != wifi_networks_.size() - 1) {
      float line_y = item_rect.y + item_height_ - 1;
      DrawLine(item_rect.x, line_y, item_rect.x + item_rect.width, line_y, RAYLIB_LIGHTGRAY);
    }
  }

  EndScissorMode();
}

void WifiManager::drawNetworkItem(const Rectangle& rect, const Network& network) {
  const int btn_width = 200;
  Rectangle label_rect{rect.x, rect.y, rect.width - btn_width * 2, item_height_};
  GuiLabel(label_rect, network.ssid.c_str());

  if (network.connected) {
    GuiLabel({rect.x + rect.width - btn_width * 2 - 20, rect.y, btn_width, item_height_}, "Connected");
  } else if (CheckCollisionPointRec(GetMousePosition(), label_rect) && IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
    initiateConnection(network.ssid);
  }

  if (saved_networks_.count(network.ssid)) {
    if (GuiButton({rect.x + rect.width - btn_width, rect.y + (item_height_ - 80) / 2, btn_width, 80}, "Forget")) {
      forgetNetwork(network.ssid);
    }
  }
}

void WifiManager::initiateConnection(const std::string& ssid) {
  if (saved_networks_.count(ssid)) {
    wifi::connect(ssid);  // Directly connect to saved network
    return;
  }

  connecting_ssid_ = ssid;
  memset(password_input_, 0, sizeof(password_input_));
  requires_password_ = true;
}

void WifiManager::forgetNetwork(const std::string& ssid) {
  wifi::forget(ssid);
  saved_networks_.erase(ssid);
  scanNetworksAsync();
}

void WifiManager::showPasswordDialog() {
  // TODO: Implement a keyboard input dialog
  int result = GuiTextInputBox(
      {GetScreenWidth() / 2.0f - 120, GetScreenHeight() / 2.0f - 60, 240, 140},
      ("Connect to " + connecting_ssid_).c_str(), "Password:", "Ok;Cancel", password_input_, 128, NULL);
  if (result < 0) return;

  if (result == 1) {
    if (wifi::connect(connecting_ssid_, password_input_)) {
      saved_networks_.insert(connecting_ssid_);
    }
  }

  connecting_ssid_.clear();
  requires_password_ = false;
}

void WifiManager::scanNetworksAsync() {
  async_task_ = std::async(std::launch::async, [this]() {
    auto networks = wifi::scan_networks();
    auto known_networks = wifi::saved_networks();

    std::unique_lock lock(mutex_);
    wifi_networks_ = std::move(networks);
    saved_networks_ = std::move(known_networks);
  });
}
