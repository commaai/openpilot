#include "system/ui/raylib/wifi_manager/wifi_manager.h"

#include <algorithm>
#include <chrono>

#include "system/ui/raylib/util.h"
#define RAYGUI_IMPLEMENTATION
#define BLANK RAYLIB_BLANK
#define RAYGUI_WINDOWBOX_STATUSBAR_HEIGHT 50
#define RAYGUI_MESSAGEBOX_BUTTON_HEIGHT 100
#define RAYGUI_MESSAGEBOX_BUTTON_PADDING 30
#define RAYGUI_TEXTINPUTBOX_BUTTON_PADDING 50
#define RAYGUI_TEXTINPUTBOX_BUTTON_HEIGHT 100
#define RAYGUI_TEXTINPUTBOX_HEIGHT 40

#include "third_party/raylib/include/raygui.h"

WifiManager::WifiManager() {
  GuiSetFont(pApp->getFont());
  GuiSetStyle(DEFAULT, TEXT_SIZE, 40);
  GuiSetStyle(DEFAULT, BACKGROUND_COLOR, ColorToInt({30, 30, 30, 255}));
  GuiSetStyle(DEFAULT, BASE_COLOR_NORMAL, ColorToInt({50, 50, 50, 255}));
  GuiSetStyle(DEFAULT, TEXT_COLOR_NORMAL, ColorToInt({200, 200, 200, 255}));

  scanNetworksAsync();
}

void WifiManager::render(const Rectangle& rect) {
  rescanIfNeeded();

  std::unique_lock lock(mutex_);

  if (current_action_ == ActionState::Forget) {
    if (!forgetNetwork()) return;
  }
  if (current_action_ == ActionState::Connect) {
    if (!connectToNetwork()) return;
  }

  renderNetworkList(rect);
}

void WifiManager::rescanIfNeeded() {
  double current_time = GetTime();
  if (current_action_ == ActionState::None && current_time - last_scan_time_ > 60.0) {
    last_scan_time_ = current_time;
    scanNetworksAsync();
  }
}

void WifiManager::renderNetworkList(const Rectangle& rect) {
  if (available_networks_.empty()) {
    GuiDrawText("Loading Wi-Fi networks...", rect, TEXT_ALIGN_CENTER, RAYLIB_WHITE);
    return;
  }

  Rectangle content_rect = {rect.x, rect.y, rect.width - 20, available_networks_.size() * item_height_};
  Rectangle scissor = {0};
  GuiScrollPanel(rect, nullptr, content_rect, &scroll_offset_, &scissor);

  BeginScissorMode(scissor.x, scissor.y, scissor.width, scissor.height);
  const int padding = 20;
  for (size_t i = 0; i < available_networks_.size(); ++i) {
    float y = content_rect.y + i * item_height_ + scroll_offset_.y;
    Rectangle item_rect = {content_rect.x + padding, y, content_rect.width - padding * 2, item_height_};

    renderNetworkItem(item_rect, available_networks_[i]);

    if (i != available_networks_.size() - 1) {
      float line_y = item_rect.y + item_height_ - 1;
      DrawLine(item_rect.x, line_y, item_rect.x + item_rect.width, line_y, RAYLIB_LIGHTGRAY);
    }
  }

  EndScissorMode();
}

void WifiManager::renderNetworkItem(const Rectangle& rect, const Network& network) {
  const int btn_width = 200;
  Rectangle label_rect{rect.x, rect.y, rect.width - btn_width * 2, item_height_};
  GuiLabel(label_rect, network.ssid.c_str());

  Rectangle state_rect = {rect.x + rect.width - btn_width * 2 - 30, rect.y, btn_width, item_height_};
  if (network.connected && current_action_ != ActionState::Connecting) {
    GuiLabel(state_rect, "Connected");
  } else if (current_action_ == ActionState::Connecting && selected_network_->ssid == network.ssid) {
    GuiLabel(state_rect, "CONNECTING...");
  } else if (current_action_ == ActionState::None &&
             CheckCollisionPointRec(GetMousePosition(), label_rect) &&
             IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
    initiateAction(network, ActionState::Connect);
  }

  if (saved_networks_.count(network.ssid)) {
    if (GuiButton({rect.x + rect.width - btn_width, rect.y + (item_height_ - 80) / 2, btn_width, 80}, "Forget") &&
        current_action_ == ActionState::None) {
      initiateAction(network, ActionState::Forget);
    }
  }
}

void WifiManager::initiateAction(const Network& network, ActionState action) {
  current_action_ = action;
  selected_network_ = network;
}

bool WifiManager::forgetNetwork() {
  int result = GuiMessageBox(
      {GetScreenWidth() / 2.0f - 512, GetScreenHeight() / 2.0f - 384, 1024, 768},
      ("Forget " + selected_network_->ssid + "?").c_str(), "Are you sure you want to forget this network?", "Yes;No");

  if (result < 0) return false;

  selected_network_.reset();
  if (result != 1) {
    current_action_ = ActionState::None;
    return true;
  }

  current_action_ = ActionState::Forgetting;
  saved_networks_.erase(selected_network_->ssid);
  for (auto& n : available_networks_) {
    if (n.ssid == selected_network_->ssid) {
      n.connected = false;
      break;
    }
  }

  async_forget_task_ = std::async(std::launch::async, [this, ssid = selected_network_->ssid]() {
    wifi::forget(ssid);
    scanNetworksAsync();
    current_action_ = ActionState::None;
  });
  return true;
}

bool WifiManager::connectToNetwork() {
  std::string password;
  if (!saved_networks_.count(selected_network_->ssid) && selected_network_->security_type != SecurityType::OPEN) {
    // TODO: Implement a software keyboard input dialog
    int result = GuiTextInputBox(
        {GetScreenWidth() / 2.0f - 512, GetScreenHeight() / 2.0f - 200, 1024, 400},
        ("Connect to " + selected_network_->ssid).c_str(), "Password:", "Ok;Cancel", password_input_buffer_, 128, NULL);
    if (result < 0) return false;

    password = password_input_buffer_;
    memset(password_input_buffer_, 0, sizeof(password_input_buffer_));
    if (result != 1) {
      selected_network_.reset();
      current_action_ = ActionState::None;
      return true;
    }
  }

  connectToNetworkAsync(selected_network_->ssid, password);
  current_action_ = ActionState::Connecting;
  return true;
}

void WifiManager::scanNetworksAsync() {
  if (async_scan_task_.valid() && async_scan_task_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
    return;
  }

  async_scan_task_ = std::async(std::launch::async, [this]() {
    auto scanned_networks = wifi::scan_networks();
    auto known_networks = wifi::saved_networks();

    std::unique_lock lock(mutex_);
    available_networks_ = std::move(scanned_networks);
    saved_networks_ = std::move(known_networks);
  });
}

void WifiManager::connectToNetworkAsync(const std::string& ssid, const std::string& password) {
  if (async_connection_task_.valid() && async_connection_task_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
    return;
  }

  async_connection_task_ = std::async(std::launch::async, [this, ssid, password]() {
    if (wifi::connect(ssid, password)) {
      std::unique_lock lock(mutex_);
      saved_networks_.insert(ssid);
      selected_network_.reset();
      current_action_ = ActionState::None;

      for (auto& network : available_networks_) {
        network.connected = (network.ssid == ssid);
      }
      std::sort(available_networks_.begin(), available_networks_.end());
    } else {
      selected_network_->security_type = SecurityType::WPA;
      current_action_ = ActionState::Connect;
    }
  });
}
