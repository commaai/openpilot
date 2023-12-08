#pragma once

#include <cstdint>
#include <vector>
#include "media/cam_sensor.h"
#include "system/camerad/cameras/camera_common.h"
#include "system/camerad/sensors/ar0231_registers.h"
#include "system/camerad/sensors/ox03c10_registers.h"

#define ANALOG_GAIN_MAX_CNT 55
const size_t FRAME_WIDTH = 1928;
const size_t FRAME_HEIGHT = 1208;
const size_t FRAME_STRIDE = 2896;  // for 12 bit output. 1928 * 12 / 8 + 4 (alignment)

class CameraInfo {
public:
  CameraInfo() = default;

  uint32_t frame_width, frame_height;
  uint32_t frame_stride;
  uint32_t frame_offset = 0;
  uint32_t extra_height = 0;
  int registers_offset = -1;
  int stats_offset = -1;

  int exposure_time_min;
  int exposure_time_max;

  float dc_gain_factor;
  int dc_gain_min_weight;
  int dc_gain_max_weight;
  float dc_gain_on_grey;
  float dc_gain_off_grey;

  float sensor_analog_gains[ANALOG_GAIN_MAX_CNT];
  int analog_gain_min_idx;
  int analog_gain_max_idx;
  int analog_gain_rec_idx;
  int analog_gain_cost_delta;
  float analog_gain_cost_low;
  float analog_gain_cost_high;
  float target_grey_factor;
  float min_ev;
  float max_ev;
};

class CameraAR0231 : public CameraInfo {
public:
  CameraAR0231();
};

class CameraOx03c10 : public CameraInfo {
public:
  CameraOx03c10();
};

void ar0231_process_registers(MultiCameraState *s, CameraState *c, cereal::FrameData::Builder &framed);
std::vector<struct i2c_random_wr_payload> ox03c10_get_exp_registers(const CameraInfo *ci, int exposure_time, int new_exp_g, bool dc_gain_enabled);
std::vector<struct i2c_random_wr_payload> ar0231_get_exp_registers(const CameraInfo *ci, int exposure_time, int new_exp_g, bool dc_gain_enabled);
