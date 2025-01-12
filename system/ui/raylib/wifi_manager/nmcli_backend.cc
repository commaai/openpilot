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

std::vector<Network> list_wifi_networks() {
  std::vector<Network> networks;
  std::string output = util::check_output("nmcli -t -f IN-USE,SSID --colors no device wifi list");
  for (const auto& line : split(output, '\n')) {
    auto items = split(line, ':');
    if (items.size() != 2 || items[1].empty()) continue;

    networks.emplace_back(Network{items[0] == "*", items[1]});
  }
  return networks;
}

bool connect_to_wifi(const std::string& ssid, const std::string& password) {
  std::string command = "nmcli device wifi connect '" + ssid + "' password '" + password + "'";
  return system(command.c_str()) == 0;
}

bool forget_wifi(const std::string& ssid) {
  std::string command = "nmcli connection delete id '" + ssid + "'";
  return system(command.c_str()) == 0;
}
