#pragma once

#include <cstdlib>
#include <fstream>
#include <map>
#include <string>

#include "cereal/messaging/messaging.h"

// no-op base hw class
class HardwareNone {
public:
  static constexpr float MAX_VOLUME = 0.7;
  static constexpr float MIN_VOLUME = 0.2;

  static std::string get_os_version() { return ""; }
  static std::string get_name() { return ""; }
  static cereal::InitData::DeviceType get_device_type() { return cereal::InitData::DeviceType::UNKNOWN; }
  static int get_voltage() { return 0; }
  static int get_current() { return 0; }

  static std::string get_serial() { return "cccccc"; }

  static std::map<std::string, std::string> get_init_logs() {
    return {};
  }

  static void reboot() {}
  static void poweroff() {}
  static void set_brightness(int percent) {}
  static void set_display_power(bool on) {}
  static void set_volume(float volume) {}

  static bool get_ssh_enabled() { return false; }
  static void set_ssh_enabled(bool enabled) {}

  static bool PC() { return false; }
  static bool TICI() { return false; }
  static bool AGNOS() { return false; }
};
