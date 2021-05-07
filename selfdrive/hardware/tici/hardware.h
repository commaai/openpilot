#pragma once

#include <cstdlib>
#include <fstream>

#include "selfdrive/common/params.h"
#include "selfdrive/common/util.h"
#include "selfdrive/hardware/base.h"

class HardwareTici : public HardwareBase {
public:
  static constexpr float MAX_VOLUME = 0.5;
  static constexpr float MIN_VOLUME = 0.4;

  static const int road_cam_focal_len = 2648;
  static const int driver_cam_focal_len = 860;
  inline static const int road_cam_size[] = {1928, 1208};
  inline static const int driver_cam_size[] = {1928, 1208};
  inline static const int wide_road_cam_size[] = {1928, 1208};
  inline static const int screen_size[] = {2160, 1080};

  static mat3 road_cam_intrinsic_matrix = (mat3){
    {road_cam_focal_len, 0.0, road_cam_size[0] / 2.0f,
    0.0, road_cam_focal_len, road_cam_size[1] / 2.0f,
    0.0, 0.0, 1.0}};

  static mat3 road_cam_intrinsic_matrix = (mat3){
      {road_cam_focal_len, 0.0, road_cam_size[0] / 2.0f,
       0.0, road_cam_focal_len, road_cam_size[1] / 2.0f,
       0.0, 0.0, 1.0}};

  static bool TICI() { return true; }
  static std::string get_os_version() {
    return "AGNOS " + util::read_file("/VERSION");
  };

  static void reboot() { std::system("sudo reboot"); };
  static void poweroff() { std::system("sudo poweroff"); };
  static void set_brightness(int percent) {
    std::ofstream brightness_control("/sys/class/backlight/panel0-backlight/brightness");
    if (brightness_control.is_open()) {
      brightness_control << (percent * (int)(1023/100.)) << "\n";
      brightness_control.close();
    }
  };
  static void set_display_power(bool on) {};

  static bool get_ssh_enabled() { return Params().getBool("SshEnabled"); };
  static void set_ssh_enabled(bool enabled) { Params().putBool("SshEnabled", enabled); };
};
