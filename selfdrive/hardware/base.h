#pragma once

#include <cstdlib>
#include <fstream>


// no-op base hw class
class HardwareNone {
public:
  static constexpr float MAX_VOLUME = 0;
  static constexpr float MIN_VOLUME = 0;

  static std::string get_os_version() { return "openpilot for PC"; };

  static void reboot() {};
  static void poweroff() {};
  static void set_brightness(int percent) {};
  static void set_display_power(bool on) {};

  static bool get_ssh_enabled() { return false; };
  static void set_ssh_enabled(bool enabled) {};
};
