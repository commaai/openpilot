#pragma once

#include <cassert>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>
#include "media/cam_sensor.h"
#include "system/camerad/cameras/camera_common.h"
#include "system/camerad/sensors/ar0231_registers.h"
#include "system/camerad/sensors/ox03c10_registers.h"
#include "system/camerad/sensors/os04c10_registers.h"

#define ANALOG_GAIN_MAX_CNT 55

class SensorInfo {
public:
  SensorInfo() = default;
  virtual std::vector<i2c_random_wr_payload> getExposureRegisters(int exposure_time, int new_exp_g, bool dc_gain_enabled) const { return {}; }
  virtual float getExposureScore(float desired_ev, int exp_t, int exp_g_idx, float exp_gain, int gain_idx) const {return 0; }
  virtual int getSlaveAddress(int port) const { assert(0); }
  virtual void processRegisters(CameraState *c, cereal::FrameData::Builder &framed) const {}

  cereal::FrameData::ImageSensor image_sensor = cereal::FrameData::ImageSensor::UNKNOWN;
  float pixel_size_mm;
  uint32_t frame_width, frame_height;
  uint32_t frame_stride;
  uint32_t frame_offset = 0;
  uint32_t extra_height = 0;
  int registers_offset = -1;
  int stats_offset = -1;
  int hdr_offset = -1;

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

  bool data_word;
  uint32_t probe_reg_addr;
  uint32_t probe_expected_data;
  std::vector<i2c_random_wr_payload> start_reg_array;
  std::vector<i2c_random_wr_payload> init_reg_array;

  uint32_t mipi_format;
  uint32_t mclk_frequency;
  uint32_t frame_data_type;
};

class AR0231 : public SensorInfo {
public:
  AR0231();
  std::vector<i2c_random_wr_payload> getExposureRegisters(int exposure_time, int new_exp_g, bool dc_gain_enabled) const override;
  float getExposureScore(float desired_ev, int exp_t, int exp_g_idx, float exp_gain, int gain_idx) const override;
  int getSlaveAddress(int port) const override;
  void processRegisters(CameraState *c, cereal::FrameData::Builder &framed) const override;

private:
  mutable std::map<uint16_t, std::pair<int, int>> ar0231_register_lut;
};

class OX03C10 : public SensorInfo {
public:
  OX03C10();
  std::vector<i2c_random_wr_payload> getExposureRegisters(int exposure_time, int new_exp_g, bool dc_gain_enabled) const override;
  float getExposureScore(float desired_ev, int exp_t, int exp_g_idx, float exp_gain, int gain_idx) const override;
  int getSlaveAddress(int port) const override;
};

class OS04C10 : public SensorInfo {
public:
  OS04C10();
  std::vector<i2c_random_wr_payload> getExposureRegisters(int exposure_time, int new_exp_g, bool dc_gain_enabled) const override;
  float getExposureScore(float desired_ev, int exp_t, int exp_g_idx, float exp_gain, int gain_idx) const override;
  int getSlaveAddress(int port) const override;
};
