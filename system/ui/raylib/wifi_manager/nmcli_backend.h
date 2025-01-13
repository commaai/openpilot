#pragma once

#include <set>
#include <string>
#include <vector>

enum class SecurityType {
  OPEN,
  WPA,
  UNSUPPORTED
};

struct Network {
  std::string ssid;
  bool connected;
  int strength;
  SecurityType security_type;
};

namespace wifi {
std::vector<Network> scan_networks();
std::set<std::string> saved_networks();
bool connect(const std::string& ssid, const std::string& password);
bool forget(const std::string& ssid);
}  // namespace wifi
