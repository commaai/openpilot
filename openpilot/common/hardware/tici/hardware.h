#pragma once

#include <cassert>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <fstream>
#include <map>
#include <string>
#include <algorithm>  // for std::clamp
#include <sys/ioctl.h>
#include <unistd.h>

#include "common/util.h"
#include "common/hardware/base.h"

class HardwareTici : public HardwareNone {
public:
  static std::optional<UfsHealth> get_ufs_health() {
    constexpr unsigned long UFS_IOCTL_QUERY = 0x5388;
    constexpr uint32_t UPIU_QUERY_OPCODE_READ_DESC = 0x1;
    constexpr uint8_t QUERY_DESC_IDN_HEALTH = 0x9;
    constexpr uint16_t QUERY_DESC_HEALTH_SIZE = 0x25;

    struct UfsQuery {
      uint32_t opcode;
      uint8_t idn;
      uint8_t reserved;
      uint16_t buf_size;
      std::array<uint8_t, QUERY_DESC_HEALTH_SIZE> buffer;
    };
    static_assert(offsetof(UfsQuery, buffer) == 8);

    UfsQuery query = {};
    query.opcode = UPIU_QUERY_OPCODE_READ_DESC;
    query.idn = QUERY_DESC_IDN_HEALTH;
    query.buf_size = QUERY_DESC_HEALTH_SIZE;

    int fd = open("/dev/sda", O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
      return std::nullopt;
    }

    int ret = ioctl(fd, UFS_IOCTL_QUERY, &query);
    close(fd);
    if (ret != 0 || query.buf_size < 5 || query.buf_size > query.buffer.size() ||
        query.buffer[0] != query.buf_size || query.buffer[1] != QUERY_DESC_IDN_HEALTH) {
      return std::nullopt;
    }

    return UfsHealth{
      query.buffer[2],
      query.buffer[3],
      query.buffer[4],
      std::vector<uint8_t>(query.buffer.begin() + 5, query.buffer.begin() + query.buf_size),
    };
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

  static void set_ir_power(int percent) {
    auto device = get_device_type();
    if (device == cereal::InitData::DeviceType::TIZI) {
      return;
    }

    int value = util::map_val(std::clamp(percent, 0, 100), 0, 100, 0, 300);
    std::ofstream("/sys/class/leds/led:switch_2/brightness") << 0 << "\n";
    std::ofstream("/sys/class/leds/led:torch_2/brightness") << value << "\n";
    std::ofstream("/sys/class/leds/led:switch_2/brightness") << value << "\n";
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
};
