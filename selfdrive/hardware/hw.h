#pragma once

#include <cstdlib>

#include <common/util.h>

#ifdef QCOM
#define Hardware HardwareEon
#endif

class HardwareBase {
public:
  virtual void reboot() = 0;
  virtual void poweroff() = 0;
  virtual void set_brightness(int percent) = 0;
};

class HardwareEon {
public:
  static std::string get_os_version() {
    return "NEOS " + util::read_file("/VERSION");
  };

  static void reboot() { std::system("reboot"); };
  static void poweroff() { std::system("LD_LIBRARY_PATH= svc power shutdown"); };
  static void set_brightness(int percent) {
    std::ofstream brightness_control("/sys/class/leds/lcd-backlight/brightness");
    if (brightness_control.is_open()) {
      brightness_control << (percent * 2.55) << "\n";
      brightness_control.close();
    }
  };
};

