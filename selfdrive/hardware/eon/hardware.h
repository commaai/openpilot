#pragma once

#include <gui/ISurfaceComposer.h>
#include <gui/SurfaceComposerClient.h>
#include <hardware/hwcomposer_defs.h>

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

  bool EON() const override { return true; }
  std::string get_os_version() const override {
    return "NEOS " + util::read_file("/VERSION");
  };

  void reboot() const override { std::system("reboot"); };
  void poweroff() const override { std::system("LD_LIBRARY_PATH= svc power shutdown"); };
  void set_brightness(int percent) const override {
    std::ofstream brightness_control("/sys/class/leds/lcd-backlight/brightness");
    if (brightness_control.is_open()) {
      brightness_control << (int)(percent * (255 / 100.)) << "\n";
      brightness_control.close();
    }
  };

  inline void set_display_power(bool on) const override {
    auto dtoken = android::SurfaceComposerClient::getBuiltInDisplay(android::ISurfaceComposer::eDisplayIdMain);
    android::SurfaceComposerClient::setDisplayPowerMode(dtoken, on ? HWC_POWER_MODE_NORMAL : HWC_POWER_MODE_OFF);
  };

  bool get_ssh_enabled() const override {
    return std::system("getprop persist.neos.ssh | grep -qF '1'") == 0;
  };
  void set_ssh_enabled(bool enabled) const override {
    std::string cmd = util::string_format("setprop persist.neos.ssh %d", enabled ? 1 : 0);
    std::system(cmd.c_str());
  };

  // android only
  bool check_activity() const {
    int ret = std::system("dumpsys SurfaceFlinger --list | grep -Fq 'com.android.settings'");
    return ret == 0;
  }

  void close_activities() const {
    if (check_activity()) {
      std::system("pm disable com.android.settings && pm enable com.android.settings");
    }
  }

  void launch_activity(std::string activity, std::string opts = "") const {
    if (!check_activity()) {
      std::string cmd = "am start -n " + activity + " " + opts +
                        " --ez extra_prefs_show_button_bar true \
                         --es extra_prefs_set_next_text ''";
      std::system(cmd.c_str());
    }
  }
  void launch_wifi() const {
    launch_activity("com.android.settings/.wifi.WifiPickerActivity", "-a android.net.wifi.PICK_WIFI_NETWORK");
  }
  void launch_tethering() const {
    launch_activity("com.android.settings/.TetherSettings");
  }
};
