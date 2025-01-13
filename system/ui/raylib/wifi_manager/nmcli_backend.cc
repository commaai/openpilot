#include "system/ui/raylib/wifi_manager/nmcli_backend.h"

#include "common/util.h"

std::vector<std::string> split(std::string_view source, char delimiter) {
  std::vector<std::string> fields;
  size_t last = 0;
  for (size_t i = 0; i < source.length(); ++i) {
    if (source[i] == delimiter) {
      fields.emplace_back(source.substr(last, i - last));
      last = i + 1;
    }
  }
  fields.emplace_back(source.substr(last));
  return fields;
}

SecurityType getSecurityType(const std::string& security) {
  if (security.empty() || security == "--") {
    return SecurityType::OPEN;
  } else if (security.find("WPA") != std::string::npos || security.find("RSN") != std::string::npos) {
    return SecurityType::WPA;
  }
  return SecurityType::UNSUPPORTED;
}

namespace wifi {

std::vector<Network> scan_networks() {
  std::vector<Network> networks;

  std::string output = util::check_output("nmcli -t -f SSID,IN-USE,SIGNAL,SECURITY --colors no device wifi list");
  for (const auto& line : split(output, '\n')) {
    auto items = split(line, ':');
    if (items.size() != 4 || items[0].empty()) continue;

    networks.emplace_back(Network{items[0], items[1] == "*", std::stoi(items[2]), getSecurityType(items[3])});
  }

  std::sort(networks.begin(), networks.end(), [](const Network& a, const Network& b) {
    return std::tie(b.connected, b.strength, b.ssid) < std::tie(a.connected, a.strength, a.ssid);
  });

  return networks;
}

std::set<std::string> saved_networks() {
  std::string cmd = "nmcli -t -f NAME,TYPE --colors no connection show | grep \":802-11-wireless\" | sed 's/^Auto //g' | cut -d':' -f1";
  auto networks = split(util::check_output(cmd), '\n');
  return std::set<std::string>(networks.begin(), networks.end());
}

bool connect(const std::string& ssid, const std::string& password) {
  std::string command = "nmcli device wifi connect '" + ssid + "' password '" + password + "'";
  return system(command.c_str()) == 0;
}

bool forget(const std::string& ssid) {
  std::string command = "nmcli connection delete id '" + ssid + "'";
  return system(command.c_str()) == 0;
}

}  // namespace wifi
