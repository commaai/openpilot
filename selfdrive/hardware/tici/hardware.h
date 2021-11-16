#pragma once

#include <cstdlib>
#include <fstream>

#include "selfdrive/common/params.h"
#include "selfdrive/common/util.h"
#include "selfdrive/hardware/base.h"

class HardwareTici : public HardwareNone {
public:
  static constexpr float MAX_VOLUME = 1.0;
  static constexpr float MIN_VOLUME = 0.4;

  static bool TICI() { return true; }
  static std::string get_os_version() {
    return "AGNOS " + util::read_file("/VERSION");
  };

  static void reboot() { std::system("sudo reboot"); };
  static void poweroff() { std::system("sudo poweroff"); };
  static void set_brightness(int percent) {
    std::ofstream brightness_control("/sys/class/backlight/panel0-backlight/brightness");
    if (brightness_control.is_open()) {
      brightness_control << (percent * (int)(1023/100.)) << "\n";
      brightness_control.close();
    }
  };
  static void set_display_power(bool on) {
    std::ofstream bl_power_control("/sys/class/backlight/panel0-backlight/bl_power");
    if (bl_power_control.is_open()) {
      bl_power_control << (on ? "0" : "4") << "\n";
      bl_power_control.close();
    }
  };

  static bool get_ssh_enabled() { return Params().getBool("SshEnabled"); };
  static void set_ssh_enabled(bool enabled) { Params().putBool("SshEnabled", enabled); };
};
