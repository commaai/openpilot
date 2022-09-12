#include "system/camerad/cameras/camera.h"

const uint32_t VS_TIME_MIN_OX03C10 = 1;
const uint32_t VS_TIME_MAX_OX03C10 = 34;  // vs < 35

// similar gain curve to AR
const float sensor_analog_gains_OX03C10[] = {
    1.0, 1.25, 1.3125, 1.5625,
    1.6875, 2.0, 2.25, 2.625,
    3.125, 3.625, 4.5, 5.0,
    7.25, 8.5, 12.0, 15.5};

CameraOX03C10::CameraOX03C10() {
  id = CAMERA_ID_AR0231;
  ci = {
      .frame_width = FRAME_WIDTH,
      .frame_height = FRAME_HEIGHT,
      .frame_stride = FRAME_STRIDE,  // (0xa80*12//8)
      .extra_height = 16,            // this right?
  };

  dc_gain_factor = 7.32;
  dc_gain_max_weight = 32;
  dc_gain_on_grey = 0.3;
  dc_gain_off_grey = 0.375;
  exposure_time_min = 2;  // 1x
  exposure_time_max = 2016;
  analog_gain_min_idx = 0x0;
  analog_gain_rec_idx = 0x5;  // 2x
  analog_gain_max_idx = 0xF;

  for (int i = 0; i <= analog_gain_max_idx; i++) {
    sensor_analog_gains[i] = sensor_analog_gains_OX03C10[i];
  }
  min_ev = (exposure_time_min + VS_TIME_MIN_OX03C10) * sensor_analog_gains[analog_gain_min_idx];
  max_ev = exposure_time_max * dc_gain_factor * sensor_analog_gains[analog_gain_max_idx];

  for (auto &v : start_reg_array_ox03c10) {
    start_reg_array.push_back(v);
  }

  for (auto &v : init_array_ox03c10) {
    init_array.push_back(v);
  }
  i2c_type = CAMERA_SENSOR_I2C_TYPE_BYTE;
  in_port_info_dt = 0x2c;  // one is 0x2a, two are 0x2b
  reg_addr = 0x300a;
  expected_data = 0x5803;
  config_val_low = 24000000;
}

CameraOX03C10::~CameraOX03C10() {
}

int CameraOX03C10::getSlaveAddress(int port) {
  assert(port >=0 && port <= 2);
  return (int[]){0x6C, 0x20, 0x6C}[port];
}

std::vector<struct i2c_random_wr_payload> CameraOX03C10::getExposureVector(int new_g, bool dc_gain_enabled, int exposure_time, int dc_gain_weight) const {
  static constexpr uint32_t ox03c10_analog_gains_reg[] = {
      0x100, 0x140, 0x150, 0x190,
      0x1B0, 0x200, 0x240, 0x2A0,
      0x320, 0x3A0, 0x480, 0x500,
      0x740, 0x880, 0xC00, 0xF80};

  uint32_t hcg_time = std::max((dc_gain_weight * exposure_time / dc_gain_max_weight), 0);
  uint32_t lcg_time = std::max(((dc_gain_max_weight - dc_gain_weight) * exposure_time / dc_gain_max_weight), 0);
  uint32_t spd_time = std::max(hcg_time / 16, (uint32_t)exposure_time_min);
  uint32_t vs_time = std::min(std::max(hcg_time / 64, VS_TIME_MIN_OX03C10), VS_TIME_MAX_OX03C10);

  uint32_t real_gain = ox03c10_analog_gains_reg[new_g];
  return {
      {0x3501, hcg_time >> 8},
      {0x3502, hcg_time & 0xFF},
      {0x3581, lcg_time >> 8},
      {0x3582, lcg_time & 0xFF},
      {0x3541, spd_time >> 8},
      {0x3542, spd_time & 0xFF},
      {0x35c1, vs_time >> 8},
      {0x35c2, vs_time & 0xFF},

      {0x3508, real_gain >> 8},
      {0x3509, real_gain & 0xFF},
      {0x3588, real_gain >> 8},
      {0x3589, real_gain & 0xFF},
      {0x3548, real_gain >> 8},
      {0x3549, real_gain & 0xFF},
      {0x35c8, real_gain >> 8},
      {0x35c9, real_gain & 0xFF},
  };
}
