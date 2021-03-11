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
    std::string val = std::to_string(percent * 2.55);
    write_file("/sys/class/leds/lcd-backlight/brightness", val.data(), val.size());
  };
};

class HardwareTici {
public:
  static std::string get_os_version() {
    return "AGNOS " + util::read_file("/VERSION");
  };

  static void reboot() { std::system("sudo reboot"); };
  static void poweroff() { std::system("sudo poweroff"); };
  static void set_brightness(int percent) {
    std::ofstream brightness_control("/sys/class/backlight/panel0-backlight/brightness");
    if (brightness_control.is_open()) {
      brightness_control << (percent * 2.55) << "\n";
      brightness_control.close();
    }
  };
};

