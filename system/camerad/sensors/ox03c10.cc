#include "system/camerad/sensors/sensor.h"

namespace {

const float DC_GAIN_OX03C10 = 7.32;

const float DC_GAIN_ON_GREY_OX03C10 = 0.9;
const float DC_GAIN_OFF_GREY_OX03C10 = 1.0;

const int DC_GAIN_MIN_WEIGHT_OX03C10 = 1;  // always on is fine
const int DC_GAIN_MAX_WEIGHT_OX03C10 = 1;

const float TARGET_GREY_FACTOR_OX03C10 = 0.01;

const float sensor_analog_gains_OX03C10[] = {
    1.0, 1.0625, 1.125, 1.1875, 1.25, 1.3125, 1.375, 1.4375, 1.5, 1.5625, 1.6875,
    1.8125, 1.9375, 2.0, 2.125, 2.25, 2.375, 2.5, 2.625, 2.75, 2.875, 3.0,
    3.125, 3.375, 3.625, 3.875, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5,
    5.75, 6.0, 6.25, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0,
    10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5};

const uint32_t ox03c10_analog_gains_reg[] = {
    0x100, 0x110, 0x120, 0x130, 0x140, 0x150, 0x160, 0x170, 0x180, 0x190, 0x1B0,
    0x1D0, 0x1F0, 0x200, 0x220, 0x240, 0x260, 0x280, 0x2A0, 0x2C0, 0x2E0, 0x300,
    0x320, 0x360, 0x3A0, 0x3E0, 0x400, 0x440, 0x480, 0x4C0, 0x500, 0x540, 0x580,
    0x5C0, 0x600, 0x640, 0x680, 0x700, 0x780, 0x800, 0x880, 0x900, 0x980, 0xA00,
    0xA80, 0xB00, 0xB80, 0xC00, 0xC80, 0xD00, 0xD80, 0xE00, 0xE80, 0xF00, 0xF80};

const int ANALOG_GAIN_MIN_IDX_OX03C10 = 0x0;
const int ANALOG_GAIN_REC_IDX_OX03C10 = 0x0;  // 1x
const int ANALOG_GAIN_MAX_IDX_OX03C10 = 0x36;
const int ANALOG_GAIN_COST_DELTA_OX03C10 = -1;
const float ANALOG_GAIN_COST_LOW_OX03C10 = 0.4;
const float ANALOG_GAIN_COST_HIGH_OX03C10 = 6.4;

const int EXPOSURE_TIME_MIN_OX03C10 = 2;  // 1x
const int EXPOSURE_TIME_MAX_OX03C10 = 2016;
const uint32_t VS_TIME_MIN_OX03C10 = 1;
const uint32_t VS_TIME_MAX_OX03C10 = 34;  // vs < 35

}  // namespace

CameraOx03c10::CameraOx03c10() {
  frame_width = FRAME_WIDTH;
  frame_height = FRAME_HEIGHT;
  frame_stride = FRAME_STRIDE;  // (0xa80*12//8)
  extra_height = 16;            // top 2 + bot 14
  frame_offset = 2;

  dc_gain_factor = DC_GAIN_OX03C10;
  dc_gain_min_weight = DC_GAIN_MIN_WEIGHT_OX03C10;
  dc_gain_max_weight = DC_GAIN_MAX_WEIGHT_OX03C10;
  dc_gain_on_grey = DC_GAIN_ON_GREY_OX03C10;
  dc_gain_off_grey = DC_GAIN_OFF_GREY_OX03C10;
  exposure_time_min = EXPOSURE_TIME_MIN_OX03C10;
  exposure_time_max = EXPOSURE_TIME_MAX_OX03C10;
  analog_gain_min_idx = ANALOG_GAIN_MIN_IDX_OX03C10;
  analog_gain_rec_idx = ANALOG_GAIN_REC_IDX_OX03C10;
  analog_gain_max_idx = ANALOG_GAIN_MAX_IDX_OX03C10;
  analog_gain_cost_delta = ANALOG_GAIN_COST_DELTA_OX03C10;
  analog_gain_cost_low = ANALOG_GAIN_COST_LOW_OX03C10;
  analog_gain_cost_high = ANALOG_GAIN_COST_HIGH_OX03C10;
  for (int i = 0; i <= analog_gain_max_idx; i++) {
    sensor_analog_gains[i] = sensor_analog_gains_OX03C10[i];
  }
  min_ev = (exposure_time_min + VS_TIME_MIN_OX03C10) * sensor_analog_gains[analog_gain_min_idx];
  max_ev = exposure_time_max * dc_gain_factor * sensor_analog_gains[analog_gain_max_idx];
  target_grey_factor = TARGET_GREY_FACTOR_OX03C10;
}

std::vector<struct i2c_random_wr_payload> ox03c10_get_exp_registers(const CameraInfo *ci, int exposure_time, int new_exp_g, bool dc_gain_enabled) {
 // t_HCG&t_LCG + t_VS on LPD, t_SPD on SPD
  uint32_t hcg_time = exposure_time;
  uint32_t lcg_time = hcg_time;
  uint32_t spd_time = std::min(std::max((uint32_t)exposure_time, (ci->exposure_time_max + VS_TIME_MAX_OX03C10) / 3), ci->exposure_time_max + VS_TIME_MAX_OX03C10);
  uint32_t vs_time = std::min(std::max((uint32_t)exposure_time / 40, VS_TIME_MIN_OX03C10), VS_TIME_MAX_OX03C10);

  uint32_t real_gain = ox03c10_analog_gains_reg[new_exp_g];

  return {
    {0x3501, hcg_time>>8}, {0x3502, hcg_time&0xFF},
    {0x3581, lcg_time>>8}, {0x3582, lcg_time&0xFF},
    {0x3541, spd_time>>8}, {0x3542, spd_time&0xFF},
    {0x35c2, vs_time&0xFF},

    {0x3508, real_gain>>8}, {0x3509, real_gain&0xFF},
  };
}
