#pragma once

#include <cassert>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "media/cam_isp.h"
#include "media/cam_sensor.h"

#include "cereal/gen/cpp/log.capnp.h"
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
  virtual void processRegisters(uint8_t *cur_buf, cereal::FrameData::Builder &framed) const {}

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

  float ev_scale = 1.0;
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

  uint32_t bits_per_pixel;
  uint32_t bayer_pattern;
  uint32_t mipi_format;
  uint32_t mclk_frequency;
  uint32_t frame_data_type;

  uint32_t readout_time_ns;  // used to recover EOF from SOF

  // ISP image processing params
  uint32_t black_level;
  std::vector<uint32_t> color_correct_matrix;  // 3x3
  std::vector<uint32_t> gamma_lut_rgb;         // gamma LUTs are length 64 * sizeof(uint32_t); same for r/g/b here
  void prepare_gamma_lut() {
    for (int i = 0; i < 64; i++) {
      gamma_lut_rgb[i] |= ((uint32_t)(gamma_lut_rgb[i+1] - gamma_lut_rgb[i]) << 10);
    }
    gamma_lut_rgb.pop_back();
  }
  std::vector<uint32_t> linearization_lut;     // length 288
  std::vector<uint32_t> linearization_pts;     // length 4
  std::vector<uint32_t> vignetting_lut;        // 2x length 884
};

class AR0231 : public SensorInfo {
public:
  AR0231();
  std::vector<i2c_random_wr_payload> getExposureRegisters(int exposure_time, int new_exp_g, bool dc_gain_enabled) const override;
  float getExposureScore(float desired_ev, int exp_t, int exp_g_idx, float exp_gain, int gain_idx) const override;
  int getSlaveAddress(int port) const override;
  void processRegisters(uint8_t *cur_buf, cereal::FrameData::Builder &framed) const override;

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
