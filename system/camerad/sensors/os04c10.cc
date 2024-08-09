#include "system/camerad/sensors/sensor.h"

namespace {

const float sensor_analog_gains_OS04C10[] = {
    1.0, 1.0625, 1.125, 1.1875, 1.25, 1.3125, 1.375, 1.4375, 1.5, 1.5625, 1.6875,
    1.8125, 1.9375, 2.0, 2.125, 2.25, 2.375, 2.5, 2.625, 2.75, 2.875, 3.0,
    3.125, 3.375, 3.625, 3.875, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5,
    5.75, 6.0, 6.25, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0,
    10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5};

const uint32_t os04c10_analog_gains_reg[] = {
    0x080, 0x088, 0x090, 0x098, 0x0A0, 0x0A8, 0x0B0, 0x0B8, 0x0C0, 0x0C8, 0x0D8,
    0x0E8, 0x0F8, 0x100, 0x110, 0x120, 0x130, 0x140, 0x150, 0x160, 0x170, 0x180,
    0x190, 0x1B0, 0x1D0, 0x1F0, 0x200, 0x220, 0x240, 0x260, 0x280, 0x2A0, 0x2C0,
    0x2E0, 0x300, 0x320, 0x340, 0x380, 0x3C0, 0x400, 0x440, 0x480, 0x4C0, 0x500,
    0x540, 0x580, 0x5C0, 0x600, 0x640, 0x680, 0x6C0, 0x700, 0x740, 0x780, 0x7C0};

}  // namespace

OS04C10::OS04C10() {
  image_sensor = cereal::FrameData::ImageSensor::OS04C10;
  pixel_size_mm = 0.004;
  data_word = false;

  hdr_offset = 64 * 2 + 8; // stagger
  frame_width = 1344;
  frame_height = 760 * 2 + hdr_offset;
  frame_stride = (frame_width * 10 / 8); // no alignment

  extra_height = 0;
  frame_offset = 0;

  start_reg_array.assign(std::begin(start_reg_array_os04c10), std::end(start_reg_array_os04c10));
  init_reg_array.assign(std::begin(init_array_os04c10), std::end(init_array_os04c10));
  probe_reg_addr = 0x300a;
  probe_expected_data = 0x5304;
  mipi_format = CAM_FORMAT_MIPI_RAW_10;
  frame_data_type = 0x2b;
  mclk_frequency = 24000000; // Hz

  dc_gain_factor = 1;
  dc_gain_min_weight = 1;  // always on is fine
  dc_gain_max_weight = 1;
  dc_gain_on_grey = 0.9;
  dc_gain_off_grey = 1.0;
  exposure_time_min = 2;
  exposure_time_max = 2400;
  analog_gain_min_idx = 0x0;
  analog_gain_rec_idx = 0x0;  // 1x
  analog_gain_max_idx = 0x36;
  analog_gain_cost_delta = -1;
  analog_gain_cost_low = 0.4;
  analog_gain_cost_high = 6.4;
  for (int i = 0; i <= analog_gain_max_idx; i++) {
    sensor_analog_gains[i] = sensor_analog_gains_OS04C10[i];
  }
  min_ev = (exposure_time_min) * sensor_analog_gains[analog_gain_min_idx];
  max_ev = exposure_time_max * dc_gain_factor * sensor_analog_gains[analog_gain_max_idx];
  target_grey_factor = 0.01;
}

std::vector<i2c_random_wr_payload> OS04C10::getExposureRegisters(int exposure_time, int new_exp_g, bool dc_gain_enabled) const {
  uint32_t long_time = exposure_time;
  uint32_t real_gain = os04c10_analog_gains_reg[new_exp_g];

  return {
    {0x3501, long_time>>8}, {0x3502, long_time&0xFF},
    {0x3508, real_gain>>8}, {0x3509, real_gain&0xFF},
    {0x350c, real_gain>>8}, {0x350d, real_gain&0xFF},
  };
}

int OS04C10::getSlaveAddress(int port) const {
  assert(port >= 0 && port <= 2);
  return (int[]){0x6C, 0x20, 0x6C}[port];
}

float OS04C10::getExposureScore(float desired_ev, int exp_t, int exp_g_idx, float exp_gain, int gain_idx) const {
  float score = std::abs(desired_ev - (exp_t * exp_gain));
  float m = exp_g_idx > analog_gain_rec_idx ? analog_gain_cost_high : analog_gain_cost_low;
  score += std::abs(exp_g_idx - (int)analog_gain_rec_idx) * m;
  score += ((1 - analog_gain_cost_delta) +
            analog_gain_cost_delta * (exp_g_idx - analog_gain_min_idx) / (analog_gain_max_idx - analog_gain_min_idx)) *
           std::abs(exp_g_idx - gain_idx) * 3.0;
  return score;
}
