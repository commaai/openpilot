#pragma once

#include <cstdlib>
#include <string>

#include "selfdrive/common/mat.h"

// no-op base hw class
class HardwareBase {
public:
  float MAX_VOLUME = 0;
  float MIN_VOLUME = 0;
  int road_cam_focal_len = 910;
  int driver_cam_focal_len = 860;
  int wide_cam_focal_len = 0;
  int road_cam_size[2] = {1164, 874};
  int driver_cam_size[2] = {1152, 864};
  int screen_size[2] = {1920, 1080};
  int wide_road_cam_size[2] = {};

  virtual std::string get_os_version() const { return ""; }
  virtual void reboot() const {}
  virtual void poweroff() const {}
  virtual void set_brightness(int percent) const {}
  virtual void set_display_power(bool on) const {}

  virtual bool get_ssh_enabled() const { return false; }
  virtual void set_ssh_enabled(bool enabled) const {}

  virtual bool PC() const { return false; }
  virtual bool EON() const { return false; }
  virtual bool TICI() const { return false; }

  inline mat3 road_cam_intrinsic_matrix() const {
    return get_matrix(road_cam_focal_len, road_cam_size);
  }
  inline mat3 driver_cam_intrinsic_matrix() const {
    return get_matrix(driver_cam_focal_len, driver_cam_size);
  }
  inline mat3 wide_cam_intrinsic_matrix() const {
    return TICI() ? get_matrix(wide_cam_focal_len, wide_road_cam_size) : (mat3){};
  }

private:
  inline mat3 get_matrix(int focal_len, const int frame_size[2]) const {
    return (mat3){{(float)focal_len, 0.0, frame_size[0] / 2.0f,
                   0.0, (float)focal_len, frame_size[1] / 2.0f,
                   0.0, 0.0, 1.0}};
  }
};
