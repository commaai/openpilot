#pragma once

#include <cassert>
#include <map>
#include <string>
#include <algorithm>  // for std::clamp

#include "common/util.h"
#include "common/hardware/base.h"

class HardwareComma : public HardwareNone {
public:
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

    // brightness b drives b / 255 of the IR LED's 300 mA, see hardware.py
    std::ofstream("/sys/class/leds/ir:torch/brightness") << std::clamp(percent, 0, 100) * 255 / 100 << "\n";
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
