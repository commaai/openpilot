#pragma once

#include "system/hardware/base.h"

class HardwarePC : public HardwareNone {
public:
  static std::string get_os_version() { return "openpilot for PC"; }
  static std::string get_name() { return "pc"; };
  static cereal::InitData::DeviceType get_device_type() { return cereal::InitData::DeviceType::PC; };
  static bool PC() { return true; }
  static bool TICI() { return util::getenv("TICI", 0) == 1; }
  static bool AGNOS() { return util::getenv("TICI", 0) == 1; }

  static void set_volume(float volume) {
    volume = util::map_val(volume, 0.f, 1.f, MIN_VOLUME, MAX_VOLUME);

    // "pactl set-sink-volume 1 0.100 &"
    char *cmd = (char *)malloc(32);
    snprintf(cmd, 32, "pactl set-sink-volume 1 %.3f &", volume);
    std::system(cmd);
    free(cmd);
  }
};
