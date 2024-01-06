#pragma once

#include <string>

#include "system/hardware/base.h"

class HardwarePC : public HardwareNone {
public:
  static std::string get_os_version() { return "openpilot for PC"; }
  static std::string get_name() { return "pc"; }
  static cereal::InitData::DeviceType get_device_type() { return cereal::InitData::DeviceType::PC; }
  static bool PC() { return true; }
  static bool TICI() { return util::getenv("TICI", 0) == 1; }
  static bool AGNOS() { return util::getenv("TICI", 0) == 1; }

  static void config_cpu_rendering(bool offscreen) {
    if (offscreen) {
      setenv("QT_QPA_PLATFORM", "offscreen", 1);
    }
    setenv("__GLX_VENDOR_LIBRARY_NAME", "mesa", 1);
    setenv("LP_NUM_THREADS", "0", 1); // disable threading so we stay on our assigned CPU
  }
};
