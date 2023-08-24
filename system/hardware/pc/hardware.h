#pragma once

#include <string>

#include "system/hardware/base.h"

class HardwarePC : public HardwareNone {
public:
  static std::string get_os_version() { return "openpilot for PC"; }
  static std::string get_name() { return "pc"; }
  static cereal::InitData::DeviceType get_device_type() { return cereal::InitData::DeviceType::PC; }
  static bool PC() { return true; }
  static bool TICI() { return util::getenv("TICI", 0) == 1; }
  static bool AGNOS() { return util::getenv("TICI", 0) == 1; }

  static void set_volume(float volume) {
    volume = util::map_val(volume, 0.f, 1.f, MIN_VOLUME, MAX_VOLUME);

    char volume_str[6];
    snprintf(volume_str, sizeof(volume_str), "%.3f", volume);
    std::system(("pactl set-sink-volume @DEFAULT_SINK@ " + std::string(volume_str)).c_str());
  }
};
