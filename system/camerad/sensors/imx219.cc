#include <cmath>

#include "system/camerad/sensors/sensor.h"

namespace {

// IMX219 analog gain = 256 / (256 - code), code 0..232
const float sensor_analog_gains_IMX219[] = {
    1.0, 1.066, 1.143, 1.231, 1.333, 1.455, 1.6, 1.778, 2.0,
    2.286, 2.667, 3.2, 4.0, 5.333, 8.0, 10.667,
};

const uint32_t imx219_analog_gains_reg[] = {
    0, 16, 32, 48, 64, 80, 96, 112, 128,
    144, 160, 176, 192, 208, 224, 232,
};

}  // namespace

IMX219::IMX219() {
  image_sensor = cereal::FrameData::ImageSensor::UNKNOWN;
  bayer_pattern = 0;  // RGGB
  pixel_size_mm = 0.00112;
  data_word = true;
  frame_width = 1920;
  frame_height = 1080;
  frame_stride = frame_width * 10 / 8;  // 2400 bytes (SRGGB10 packed)
  extra_height = 0;
  frame_offset = 0;

  // kernel IMX219 driver handles init and start — empty arrays
  probe_reg_addr = 0x0000;
  probe_expected_data = 0x0219;
  bits_per_pixel = 10;
  mipi_format = 0;
  frame_data_type = 0;
  mclk_frequency = 24000000;

  readout_time_ns = 10000000;  // ~10ms for 1080p

  dc_gain_factor = 1.0;
  dc_gain_min_weight = 0;
  dc_gain_max_weight = 0;
  dc_gain_on_grey = 0.9;
  dc_gain_off_grey = 1.0;
  exposure_time_min = 4;
  exposure_time_max = 1759;
  analog_gain_min_idx = 0;
  analog_gain_rec_idx = 0;
  analog_gain_max_idx = 15;
  analog_gain_cost_delta = -1;
  analog_gain_cost_low = 0.4;
  analog_gain_cost_high = 6.4;
  for (int i = 0; i <= analog_gain_max_idx; i++) {
    sensor_analog_gains[i] = sensor_analog_gains_IMX219[i];
  }
  min_ev = exposure_time_min * sensor_analog_gains[analog_gain_min_idx];
  max_ev = exposure_time_max * sensor_analog_gains[analog_gain_max_idx];
  target_grey_factor = 0.01;
  ev_scale = 1.0;

  black_level = 64;  // IMX219 10-bit black level
  color_correct_matrix = {
    0x00000080, 0x00000000, 0x00000000,
    0x00000000, 0x00000080, 0x00000000,
    0x00000000, 0x00000000, 0x00000080,
  };
  for (int i = 0; i < 65; i++) {
    float fx = i / 64.0f;
    fx = pow(fx, 1.0f / 2.2f);
    gamma_lut_rgb.push_back((uint32_t)(fx * 1023.0f + 0.5f));
  }
  prepare_gamma_lut();

  // identity linearization (IMX219 is linear, no correction needed)
  linearization_lut.assign(36, 0x00000000);
  linearization_pts = {0x00000000, 0x00000000, 0x00000000, 0x00000000};

  // no vignetting correction — fill with unity (1.0 in Q1.12 = 0x00800000)
  vignetting_lut.assign(221, 0x00800000);
}

std::vector<i2c_random_wr_payload> IMX219::getExposureRegisters(int exposure_time, int new_exp_g, bool dc_gain_enabled) const {
  // IMX219 exposure: 16-bit register at 0x015A-0x015B
  // IMX219 analog gain: 8-bit register at 0x0157
  uint32_t gain_code = imx219_analog_gains_reg[new_exp_g];
  return {
    {0x015A, (uint32_t)((exposure_time >> 8) & 0xFF)},
    {0x015B, (uint32_t)(exposure_time & 0xFF)},
    {0x0157, gain_code},
  };
}

float IMX219::getExposureScore(float desired_ev, int exp_t, int exp_g_idx, float exp_gain, int gain_idx) const {
  float score = std::abs(desired_ev - (exp_t * exp_gain));
  float m = exp_g_idx > gain_idx ? analog_gain_cost_low : analog_gain_cost_high;
  score += std::abs(exp_g_idx - gain_idx) * m;
  return score;
}

int IMX219::getSlaveAddress(int port) const {
  return 0x10;
}
