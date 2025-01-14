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
  std::string output = util::check_output("nmcli -t -c no -f SSID,IN-USE,SIGNAL,SECURITY device wifi list");

  for (const auto& line : split(output, '\n')) {
    auto fields = split(line, ':');
    if (fields.size() == 4 && !fields[0].empty()) {
      networks.emplace_back(Network{fields[0], fields[1] == "*", std::stoi(fields[2]), getSecurityType(fields[3])});
    }
  }

  std::sort(networks.begin(), networks.end());
  return networks;
}

std::set<std::string> saved_networks() {
  std::string uuids;
  std::string cmd = "nmcli -t -f UUID,TYPE connection show | grep 802-11-wireless";
  for (auto& line : split(util::check_output(cmd), '\n')) {
    auto connection_info = split(line, ':');
    if (connection_info.size() >= 2) {
      uuids += connection_info[0] + " ";
    }
  }

  std::set<std::string> network_ssids;
  std::string ssid_cmd = "nmcli -t -f 802-11-wireless.ssid connection show " + uuids;
  for (const auto& line : split(util::check_output(ssid_cmd), '\n')) {
    if (!line.empty()) {
      network_ssids.insert(split(line, ':')[1]);
    }
  }
  return network_ssids;
}

bool connect(const std::string& ssid, const std::string& password) {
  std::string command = "nmcli device wifi connect '" + ssid + "'";
  if (!password.empty()) {
    command += " password '" + password + "'";
  }
  return system(command.c_str()) == 0;
}

bool forget(const std::string& ssid) {
  std::string command = "nmcli connection delete id '" + ssid + "'";
  return system(command.c_str()) == 0;
}

}  // namespace wifi
