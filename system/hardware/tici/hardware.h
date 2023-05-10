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

  static std::string get_name() {
    std::string devicetree_model = util::read_file("/sys/firmware/devicetree/base/model");
    return (devicetree_model.find("tizi") != std::string::npos) ? "tizi" : "tici";
  };

  static cereal::InitData::DeviceType get_device_type() {
    return (get_name() == "tizi") ? cereal::InitData::DeviceType::TIZI : cereal::InitData::DeviceType::TICI;
  };

  static int get_voltage() { return std::atoi(util::read_file("/sys/class/hwmon/hwmon1/in1_input").c_str()); };
  static int get_current() { return std::atoi(util::read_file("/sys/class/hwmon/hwmon1/curr1_input").c_str()); };

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

  static void reboot() { std::system("sudo reboot"); };
  static void poweroff() { std::system("sudo poweroff"); };
  static void set_brightness(int percent) {
    std::string max = util::read_file("/sys/class/backlight/panel0-backlight/max_brightness");

    std::ofstream brightness_control("/sys/class/backlight/panel0-backlight/brightness");
    if (brightness_control.is_open()) {
      brightness_control << (int)(percent * (std::stof(max)/100.)) << "\n";
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


  static std::map<std::string, std::string> get_init_logs() {
    std::map<std::string, std::string> ret = {
      {"/BUILD", util::read_file("/BUILD")},
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

  static bool get_ssh_enabled() { return Params().getBool("SshEnabled"); };
  static void set_ssh_enabled(bool enabled) { Params().putBool("SshEnabled", enabled); };
};
