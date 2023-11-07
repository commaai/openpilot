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

  static void config_cpu_rendering() {
    setenv("QT_QPA_PLATFORM", "offscreen", 1);
    setenv("__GLX_VENDOR_LIBRARY_NAME", "mesa", 1);
    setenv("LP_NUM_THREADS", "0", 1); // disable threading so we stay on our assigned CPU
  }
};
