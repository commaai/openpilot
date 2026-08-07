#pragma once

#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "openpilot/cereal/gen/cpp/log.capnp.h"

// no-op base hw class
class HardwareNone {
public:
  struct UfsHealth {
    uint8_t pre_eol_info;
    uint8_t life_time_estimate_a;
    uint8_t life_time_estimate_b;
    std::vector<uint8_t> vendor_health_report;
  };

  static std::string get_name() { return ""; }
  static cereal::InitData::DeviceType get_device_type() { return cereal::InitData::DeviceType::UNKNOWN; }

  static std::string get_serial() { return "cccccc"; }

  static std::map<std::string, std::string> get_init_logs() {
    return {};
  }

  static std::optional<UfsHealth> get_ufs_health() { return std::nullopt; }

  static void set_ir_power(int percentage) {}

  static bool PC() { return false; }
};
