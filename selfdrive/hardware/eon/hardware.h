#pragma once

#include <hardware/hwcomposer_defs.h>
#ifndef _USING_LIBCXX
#define _USING_LIBCXX
#endif
#include <gui/ISurfaceComposer.h>
#include <gui/SurfaceComposerClient.h>

#include <cstdlib>
#include <fstream>

#include "selfdrive/common/util.h"
#include "selfdrive/hardware/base.h"

class Hardware : public HardwareBase {
public:
  Hardware() {
    MAX_VOLUME = 1.0;
    MIN_VOLUME = 0.5;
  }

  inline bool EON() override { return true; }

  inline std::string get_os_version() override {
    return "NEOS " + util::read_file("/VERSION");
  }

  void reboot() override {
    std::system("reboot");
  }

  void poweroff() override {
    std::system("LD_LIBRARY_PATH= svc power shutdown");
  }

  void set_brightness(int percent) override {
    std::ofstream brightness_control("/sys/class/leds/lcd-backlight/brightness");
    if (brightness_control.is_open()) {
      brightness_control << (int)(percent * (255 / 100.)) << "\n";
      brightness_control.close();
    }
  }

  static inline void set_display_power(bool on) {
    auto dtoken = android::SurfaceComposerClient::getBuiltInDisplay(android::ISurfaceComposer::eDisplayIdMain);
    android::SurfaceComposerClient::setDisplayPowerMode(dtoken, on ? HWC_POWER_MODE_NORMAL : HWC_POWER_MODE_OFF);
  }

  bool get_ssh_enabled() override {
    return std::system("getprop persist.neos.ssh | grep -qF '1'") == 0;
  }

  void set_ssh_enabled(bool enabled) override {
    std::string cmd = util::string_format("setprop persist.neos.ssh %d", enabled ? 1 : 0);
    std::system(cmd.c_str());
  }

  // android only
  void check_activity() {
    int ret = std::system("dumpsys SurfaceFlinger --list | grep -Fq 'com.android.settings'");
    launched_activity = ret == 0;
  }

  void close_activities() {
    if (launched_activity) {
      std::system("pm disable com.android.settings && pm enable com.android.settings");
    }
  }

  void launch_activity(std::string activity, std::string opts = "") {
    if (!launched_activity) {
      std::string cmd = "am start -n " + activity + " " + opts +
                        " --ez extra_prefs_show_button_bar true \
                         --es extra_prefs_set_next_text ''";
      std::system(cmd.c_str());
    }
    launched_activity = true;
  }

  void launch_wifi() {
    launch_activity("com.android.settings/.wifi.WifiPickerActivity", "-a android.net.wifi.PICK_WIFI_NETWORK");
  }

  void launch_tethering() {
    launch_activity("com.android.settings/.TetherSettings");
  }

  bool launched_activity = false;
};
