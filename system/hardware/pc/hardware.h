#pragma once

#include <string>

#include "system/hardware/base.h"

class HardwarePC : public HardwareNone {
public:
  static std::string get_os_version() { return "openpilot for PC"; }
  static std::string get_name() { return "pc"; }
  static cereal::InitData::DeviceType get_device_type() { return cereal::InitData::DeviceType::PC; }
  static bool PC() { return true; }
  static bool IS_TICI() { return util::getenv("TICI", 0) == 1; }
};
