#pragma once

#include <cstdlib>
#include <fstream>
#include <map>
#include <string>

#include "openpilot/cereal/gen/cpp/log.capnp.h"

// no-op base hw class
class HardwareNone {
public:
  static std::string get_name() { return ""; }
  static cereal::InitData::DeviceType get_device_type() { return cereal::InitData::DeviceType::UNKNOWN; }

  static std::string get_serial() { return "cccccc"; }

  static std::map<std::string, std::string> get_init_logs() {
    return {};
  }

  static void set_ir_power(int percentage) {}

  static bool PC() { return false; }
};
