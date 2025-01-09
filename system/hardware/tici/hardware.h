#pragma once

#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <algorithm>  // for std::clamp

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
  }

  static std::string get_name() {
    std::string model = util::read_file("/sys/firmware/devicetree/base/model");
    return model.substr(std::string("comma ").size());
  }

  static cereal::InitData::DeviceType get_device_type() {
    static const std::map<std::string, cereal::InitData::DeviceType> device_map = {
      {"tici", cereal::InitData::DeviceType::TICI},
      {"tizi", cereal::InitData::DeviceType::TIZI},
      {"mici", cereal::InitData::DeviceType::MICI}
    };
    auto it = device_map.find(get_name());
    return it != device_map.end() ? it->second : cereal::InitData::DeviceType::UNKNOWN;
  }

  static int get_voltage() { return std::atoi(util::read_file("/sys/class/hwmon/hwmon1/in1_input").c_str()); }
  static int get_current() { return std::atoi(util::read_file("/sys/class/hwmon/hwmon1/curr1_input").c_str()); }

  static std::string get_serial() {
    static std::string serial("");
    if (serial.empty()) {
      std::ifstream stream("/proc/cmdline");
      std::string cmdline;
      std::getline(stream, cmdline);

      auto start = cmdline.find("serialno=");
      if (start == std::string::npos) {
        serial = "cccccc";
      } else {
        auto end = cmdline.find(" ", start + 9);
        serial = cmdline.substr(start + 9, end - start - 9);
      }
    }
    return serial;
  }

  static void reboot() { std::system("sudo reboot"); }
  static void poweroff() { std::system("sudo poweroff"); }
  static void set_brightness(int percent) {
    std::string max = util::read_file("/sys/class/backlight/panel0-backlight/max_brightness");

    std::ofstream brightness_control("/sys/class/backlight/panel0-backlight/brightness");
    if (brightness_control.is_open()) {
      brightness_control << (int)(percent * (std::stof(max)/100.)) << "\n";
      brightness_control.close();
    }
  }
  static void set_display_power(bool on) {
    std::ofstream bl_power_control("/sys/class/backlight/panel0-backlight/bl_power");
    if (bl_power_control.is_open()) {
      bl_power_control << (on ? "0" : "4") << "\n";
      bl_power_control.close();
    }
  }

  static void set_ir_power(int percent) {
    auto device = get_device_type();
    if (device == cereal::InitData::DeviceType::TICI ||
        device == cereal::InitData::DeviceType::TIZI) {
      return;
    }

    percent = std::clamp(percent, 0, 100);

    std::ofstream torch_brightness("/sys/class/leds/led:torch_2/brightness");
    if (torch_brightness.is_open()) {
      torch_brightness << percent << "\n";
      torch_brightness.close();
    }

    std::ofstream switch_brightness("/sys/class/leds/led:switch_2/brightness");
    if (switch_brightness.is_open()) {
      switch_brightness << percent << "\n";
      switch_brightness.close();
    }
  }

  static std::map<std::string, std::string> get_init_logs() {
    std::map<std::string, std::string> ret = {
      {"/BUILD", util::read_file("/BUILD")},
      {"lsblk", util::check_output("lsblk -o NAME,SIZE,STATE,VENDOR,MODEL,REV,SERIAL")},
      {"SOM ID", util::read_file("/sys/devices/platform/vendor/vendor:gpio-som-id/som_id")},
    };

    std::string bs = util::check_output("abctl --boot_slot");
    ret["boot slot"] = bs.substr(0, bs.find_first_of("\n"));

    std::string temp = util::read_file("/dev/disk/by-partlabel/ssd");
    temp.erase(temp.find_last_not_of(std::string("\0\r\n", 3))+1);
    ret["boot temp"] = temp;

    // TODO: log something from system and boot
    for (std::string part : {"xbl", "abl", "aop", "devcfg", "xbl_config"}) {
      for (std::string slot : {"a", "b"}) {
        std::string partition = part + "_" + slot;
        std::string hash = util::check_output("sha256sum /dev/disk/by-partlabel/" + partition);
        ret[partition] = hash.substr(0, hash.find_first_of(" "));
      }
    }

    return ret;
  }

  static bool get_ssh_enabled() { return Params().getBool("SshEnabled"); }
  static void set_ssh_enabled(bool enabled) { Params().putBool("SshEnabled", enabled); }

  static void config_cpu_rendering(bool offscreen) {
    if (offscreen) {
      setenv("QT_QPA_PLATFORM", "eglfs", 1); // offscreen doesn't work with EGL/GLES
    }
    setenv("LP_NUM_THREADS", "0", 1); // disable threading so we stay on our assigned CPU
  }
};
