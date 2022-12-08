#pragma once

#include <cstdlib>
#include <fstream>

#include "common/params.h"
#include "common/util.h"
#include "system/hardware/base.h"

class HardwareTici : public HardwareNone {
public:
  static constexpr float MAX_VOLUME = 0.9;
  static constexpr float MIN_VOLUME = 0.1;
  static bool TICI() { return true; }
  static bool AGNOS() { return true; }
  static std::string get_os_version() {
    return "AGNOS " + util::read_file("/VERSION");
  };
  static std::string get_name() { return "tici"; };
  static cereal::InitData::DeviceType get_device_type() { return cereal::InitData::DeviceType::TICI; };
  static int get_voltage() { return std::atoi(util::read_file("/sys/class/hwmon/hwmon1/in1_input").c_str()); };
  static int get_current() { return std::atoi(util::read_file("/sys/class/hwmon/hwmon1/curr1_input").c_str()); };


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
  static void set_volume(float volume) {
    volume = util::map_val(volume, 0.f, 1.f, MIN_VOLUME, MAX_VOLUME);

    char volume_str[6];
    snprintf(volume_str, sizeof(volume_str), "%.3f", volume);
    std::system(("pactl set-sink-volume @DEFAULT_SINK@ " + std::string(volume_str)).c_str());
  }

  static bool get_ssh_enabled() { return Params().getBool("SshEnabled"); };
  static void set_ssh_enabled(bool enabled) { Params().putBool("SshEnabled", enabled); };
};
