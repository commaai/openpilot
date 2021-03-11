#pragma once

#include <cstdlib>
#include <fstream>

#include <common/util.h>

#ifdef QCOM
#define Hardware HardwareEon
#elif QCOM2
#define Hardware HardwareTici
#else
#define Hardware HardwareNone
#endif


// no-op base hw class
class HardwareNone {
public:
  static std::string get_os_version() { return "openpilot for PC"; };

  static void reboot() {};
  static void poweroff() {};
  static void set_brightness(int percent) {};
};

class HardwareEon : public HardwareNone {
public:
  static std::string get_os_version() {
    return "NEOS " + util::read_file("/VERSION");
  };

  static void reboot() { std::system("reboot"); };
  static void poweroff() { std::system("LD_LIBRARY_PATH= svc power shutdown"); };
  static void set_brightness(int percent) {
    std::ofstream brightness_control("/sys/class/leds/lcd-backlight/brightness");
    if (brightness_control.is_open()) {
      brightness_control << (int)(percent * (255/100.)) << "\n";
      brightness_control.close();
    }
  };
};

class HardwareTici : public HardwareNone {
public:
  static std::string get_os_version() {
    return "AGNOS " + util::read_file("/VERSION");
  };

  static void reboot() { std::system("sudo reboot"); };
  static void poweroff() { std::system("sudo poweroff"); };
  static void set_brightness(int percent) {
    std::ofstream brightness_control("/sys/class/backlight/panel0-backlight/brightness");
    if (brightness_control.is_open()) {
      brightness_control << (percent * (1023/100.)) << "\n";
      brightness_control.close();
    }
  };
};

