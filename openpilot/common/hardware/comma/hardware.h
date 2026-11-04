#pragma once

#include <cassert>
#include <map>
#include <string>
#include <algorithm>  // for std::clamp
#include <sys/utsname.h>

#include "common/util.h"
#include "common/hardware/base.h"

class HardwareComma : public HardwareNone {
public:
  // the same check as common/kernel.py: AGNOS's downstream kernel is 4.9
  static bool upstream_kernel() {
    static const bool upstream = [] {
      struct utsname u;
      return util::file_exists("/AGNOS") && uname(&u) == 0 && atoi(u.release) > 4;
    }();
    return upstream;
  }

  // the upstream kernel's IR LED, see hardware.py. the downstream kernel names it led:torch_2
  inline static const std::string IR_TORCH = "/sys/class/leds/ir:torch";
  static constexpr int IR_TORCH_MAX_MA = 300;
  static constexpr int LED_FULL = 255;

  // the ir:torch brightness for the current the downstream kernel drives at this percentage, see hardware.py
  static int ir_torch_brightness(int percent) {
    int64_t ma = util::map_val(std::clamp(percent, 0, 100), 0, 100, 0, 300);
    if (ma == 0) return 0;
    int64_t ua = std::max<int64_t>((ma * 1000 + 6250) / 12500, 1) * 12500;
    int64_t div = IR_TORCH_MAX_MA * 1000;
    return std::min<int64_t>((ua * LED_FULL + div - 1) / div, LED_FULL);
  }

  static std::string get_name() {
    static const std::string name = []() {
      std::string model = util::read_file("/sys/firmware/devicetree/base/model");
      return util::strip(model.substr(std::string("comma ").size()));
    }();
    return name;
  }

  static cereal::InitData::DeviceType get_device_type() {
    static const std::map<std::string, cereal::InitData::DeviceType> device_map = {
      {"tizi", cereal::InitData::DeviceType::TIZI},
      {"mici", cereal::InitData::DeviceType::MICI}
    };
    static const auto it = device_map.find(get_name());
    assert(it != device_map.end());
    return it->second;
  }

  static void set_ir_power(int percent) {
    auto device = get_device_type();
    if (device == cereal::InitData::DeviceType::TIZI) {
      return;
    }

    if (upstream_kernel()) {
      std::ofstream(IR_TORCH + "/brightness") << ir_torch_brightness(percent) << "\n";
      return;
    }

    int value = util::map_val(std::clamp(percent, 0, 100), 0, 100, 0, 300);
    std::ofstream("/sys/class/leds/led:switch_2/brightness") << 0 << "\n";
    std::ofstream("/sys/class/leds/led:torch_2/brightness") << value << "\n";
    std::ofstream("/sys/class/leds/led:switch_2/brightness") << value << "\n";
  }

  static std::map<std::string, std::string> get_init_logs(bool route_log = false) {
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

    // TODO: these are too slow to do on route log inits. need to do it async?
    if (!route_log) {
      for (std::string part : {"xbl", "abl", "aop", "devcfg", "xbl_config"}) {
        for (std::string slot : {"a", "b"}) {
          std::string partition = part + "_" + slot;
          std::string hash = util::check_output("sha256sum /dev/disk/by-partlabel/" + partition);
          ret[partition] = hash.substr(0, hash.find_first_of(" "));
        }
      }
    }

    return ret;
  }
};
