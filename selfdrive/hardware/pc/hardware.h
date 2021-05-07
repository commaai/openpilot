#pragma once

#include <cstdlib>

#include "selfdrive/common/mat.h"
#include "selfdrive/hardware/base.h"

class HardwarePC : public HardwareBase {
public:
  static const int road_cam_focal_len = 910;
  static const int driver_cam_focal_len = 860;
  inline static const int road_cam_size[] = {1164, 874};
  inline static const int driver_cam_size[] = {1152, 864};
  inline static const int screen_size[] = {1920, 1080};
  inline static mat3 road_cam_intrinsic_matrix = (mat3){
    {road_cam_focal_len, 0.0, road_cam_size[0] / 2.0f,
    0.0, road_cam_focal_len, road_cam_size[1] / 2.0f,
    0.0, 0.0, 1.0}};

  static std::string get_os_version() { return "openpilot for PC"; }
  static bool PC() { return true; }
};
