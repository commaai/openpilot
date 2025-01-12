#pragma once

#include <string>
#include <vector>

struct Network {
  bool connected;
  std::string ssid;
};

std::vector<Network> list_wifi_networks();
bool connect_to_wifi(const std::string& ssid, const std::string& password);
bool forget_wifi(const std::string& ssid);
