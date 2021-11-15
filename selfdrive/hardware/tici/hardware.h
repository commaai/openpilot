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
  static constexpr float SECONDS_IN_HOUR = 60*60;
  static constexpr float HOURLY_BRIGHTNESS_DECREASE = 5;
  static constexpr float BRIGHTNESS_LIMIT_MIN = 30;
  static constexpr float BRIGHTNESS_LIMIT_MAX = 100;
  static constexpr float MAX_BRIGHTNESS_HOURS = 4;

  static bool TICI() { return true; }
  static std::string get_os_version() {
    return "AGNOS " + util::read_file("/VERSION");
  };

  static void reboot() { std::system("sudo reboot"); };
  static void poweroff() { std::system("sudo poweroff"); };
  static void set_brightness(int percent, float ui_running_time) {
    float ui_running_hours = ui_running_time / SECONDS_IN_HOUR;
    int anti_burnin_max_percent = std::clamp((int) (BRIGHTNESS_LIMIT_MAX -  HOURLY_BRIGHTNESS_DECREASE * (ui_running_hours - MAX_BRIGHTNESS_HOURS)),
                                             BRIGHTNESS_LIMIT_MIN,
                                             BRIGHTNESS_LIMIT_MIN);
    percent = std::min(percent, anti_burnin_max_percent);
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
