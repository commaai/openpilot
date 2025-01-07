#include <cmath>

#include "system/camerad/sensors/sensor.h"

namespace {

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

const uint32_t VS_TIME_MIN_OX03C10 = 1;
const uint32_t VS_TIME_MAX_OX03C10 = 34;  // vs < 35

}  // namespace

OX03C10::OX03C10() {
  image_sensor = cereal::FrameData::ImageSensor::OX03C10;
  bayer_pattern = CAM_ISP_PATTERN_BAYER_GRGRGR;
  pixel_size_mm = 0.003;
  data_word = false;
  frame_width = 1928;
  frame_height = 1208;
  frame_stride = (frame_width * 12 / 8) + 4;
  extra_height = 16;            // top 2 + bot 14
  frame_offset = 2;

  start_reg_array.assign(std::begin(start_reg_array_ox03c10), std::end(start_reg_array_ox03c10));
  init_reg_array.assign(std::begin(init_array_ox03c10), std::end(init_array_ox03c10));
  probe_reg_addr = 0x300a;
  probe_expected_data = 0x5803;
  bits_per_pixel = 12;
  mipi_format = CAM_FORMAT_MIPI_RAW_12;
  frame_data_type = 0x2c; // one is 0x2a, two are 0x2b
  mclk_frequency = 24000000; //Hz

  readout_time_ns = 14697000;

  dc_gain_factor = 7.32;
  dc_gain_min_weight = 1;  // always on is fine
  dc_gain_max_weight = 1;
  dc_gain_on_grey = 0.9;
  dc_gain_off_grey = 1.0;
  exposure_time_min = 2;  // 1x
  exposure_time_max = 2016;
  analog_gain_min_idx = 0x0;
  analog_gain_rec_idx = 0x0;  // 1x
  analog_gain_max_idx = 0x36;
  analog_gain_cost_delta = -1;
  analog_gain_cost_low = 0.4;
  analog_gain_cost_high = 6.4;
  for (int i = 0; i <= analog_gain_max_idx; i++) {
    sensor_analog_gains[i] = sensor_analog_gains_OX03C10[i];
  }
  min_ev = (exposure_time_min + VS_TIME_MIN_OX03C10) * sensor_analog_gains[analog_gain_min_idx];
  max_ev = exposure_time_max * dc_gain_factor * sensor_analog_gains[analog_gain_max_idx];
  target_grey_factor = 0.01;

  black_level = 0;
  color_correct_matrix = {
    0x000000b6, 0x00000ff1, 0x00000fda,
    0x00000fcc, 0x000000b9, 0x00000ffb,
    0x00000fc2, 0x00000ff6, 0x000000c9,
  };
  for (int i = 0; i < 65; i++) {
    float fx = i / 64.0;
    fx = -0.507089*exp(-12.54124638*fx) + 0.9655*pow(fx, 0.5) - 0.472597*fx + 0.507089;
    gamma_lut_rgb.push_back((uint32_t)(fx*1023.0 + 0.5));
  }
  prepare_gamma_lut();
  linearization_lut = {
    0x00200000, 0x00200000, 0x00200000, 0x00200000,
    0x00404080, 0x00404080, 0x00404080, 0x00404080,
    0x00804100, 0x00804100, 0x00804100, 0x00804100,
    0x02014402, 0x02014402, 0x02014402, 0x02014402,
    0x0402c804, 0x0402c804, 0x0402c804, 0x0402c804,
    0x0805d00a, 0x0805d00a, 0x0805d00a, 0x0805d00a,
    0x100ba015, 0x100ba015, 0x100ba015, 0x100ba015,
    0x00003fff, 0x00003fff, 0x00003fff, 0x00003fff,
    0x00003fff, 0x00003fff, 0x00003fff, 0x00003fff,
  };
  linearization_pts = {0x07ff0bff, 0x17ff1bff, 0x1fff23ff, 0x27ff3fff};
  vignetting_lut = {
    0x00eaa755, 0x00cf2679, 0x00bc05e0, 0x00acc566, 0x00a1450a, 0x009984cc, 0x0095a4ad, 0x009584ac, 0x009944ca, 0x00a0c506, 0x00ac0560, 0x00bb25d9, 0x00ce2671, 0x00e90748, 0x01112889, 0x014a2a51, 0x01984cc2,
    0x00db06d8, 0x00c30618, 0x00afe57f, 0x00a0a505, 0x009524a9, 0x008d646b, 0x0089844c, 0x0089644b, 0x008d2469, 0x0094a4a5, 0x009fe4ff, 0x00af0578, 0x00c20610, 0x00d986cc, 0x00fda7ed, 0x01320990, 0x017aebd7,
    0x00d1868c, 0x00baa5d5, 0x00a7853c, 0x009844c2, 0x008cc466, 0x0085a42d, 0x0083641b, 0x0083641b, 0x0085842c, 0x008c4462, 0x0097a4bd, 0x00a6c536, 0x00b9a5cd, 0x00d06683, 0x00f1678b, 0x01226913, 0x0167ab3d,
    0x00cd0668, 0x00b625b1, 0x00a30518, 0x0093c49e, 0x00884442, 0x00830418, 0x0080e407, 0x0080c406, 0x0082e417, 0x0087c43e, 0x00932499, 0x00a22511, 0x00b525a9, 0x00cbe65f, 0x00eb0758, 0x011a68d3, 0x015daaed,
    0x00cc4662, 0x00b565ab, 0x00a24512, 0x00930498, 0x0087843c, 0x0082a415, 0x00806403, 0x00806403, 0x00828414, 0x00870438, 0x00926493, 0x00a1850c, 0x00b465a3, 0x00cb2659, 0x00ea2751, 0x011928c9, 0x015c2ae1,
    0x00cf667b, 0x00b885c4, 0x00a5652b, 0x009624b1, 0x008aa455, 0x00846423, 0x00822411, 0x00822411, 0x00844422, 0x008a2451, 0x009564ab, 0x00a48524, 0x00b785bc, 0x00ce4672, 0x00ee6773, 0x011e88f4, 0x0162eb17,
    0x00d6c6b6, 0x00bf65fb, 0x00ac4562, 0x009d04e8, 0x0091848c, 0x0089c44e, 0x00862431, 0x00860430, 0x0089844c, 0x00910488, 0x009c64e3, 0x00ab655b, 0x00be65f3, 0x00d566ab, 0x00f847c2, 0x012b2959, 0x01726b93,
    0x00e3e71f, 0x00ca0650, 0x00b705b8, 0x00a7a53d, 0x009c24e1, 0x009484a4, 0x00908484, 0x00908484, 0x009424a1, 0x009bc4de, 0x00a70538, 0x00b625b1, 0x00c90648, 0x00e26713, 0x0108e847, 0x013fe9ff, 0x018bcc5e,
    0x00f807c0, 0x00d966cb, 0x00c5862c, 0x00b625b1, 0x00aaa555, 0x00a30518, 0x009f04f8, 0x009f04f8, 0x00a2a515, 0x00aa2551, 0x00b585ac, 0x00c4a625, 0x00d846c2, 0x00f647b2, 0x0121a90d, 0x015e4af2, 0x01b8cdc6,
    0x011548aa, 0x00f1678b, 0x00d886c4, 0x00c86643, 0x00bce5e7, 0x00b545aa, 0x00b1658b, 0x00b1458a, 0x00b505a8, 0x00bc85e4, 0x00c7c63e, 0x00d786bc, 0x00efe77f, 0x0113489a, 0x0144ea27, 0x01888c44, 0x01fdcfee,
    0x013e49f2, 0x0113e89f, 0x00f5a7ad, 0x00e0c706, 0x00d30698, 0x00cb665b, 0x00c7663b, 0x00c7663b, 0x00cb0658, 0x00d2a695, 0x00dfe6ff, 0x00f467a3, 0x01122891, 0x013be9df, 0x01750ba8, 0x01cfae7d, 0x025912c8,
    0x01766bb3, 0x01446a23, 0x011fc8fe, 0x0105e82f, 0x00f467a3, 0x00e9874c, 0x00e46723, 0x00e44722, 0x00e92749, 0x00f3a79d, 0x0104c826, 0x011e48f2, 0x01424a12, 0x01738b9c, 0x01bf6dfb, 0x023611b0, 0x02ced676,
    0x01cf8e7c, 0x01866c33, 0x015aaad5, 0x013ae9d7, 0x01250928, 0x011768bb, 0x0110a885, 0x01108884, 0x0116e8b7, 0x01242921, 0x0139a9cd, 0x0158eac7, 0x01840c20, 0x01cb0e58, 0x0233719b, 0x02b9d5ce, 0x03645b22,
  };
}

std::vector<i2c_random_wr_payload> OX03C10::getExposureRegisters(int exposure_time, int new_exp_g, bool dc_gain_enabled) const {
 // t_HCG&t_LCG + t_VS on LPD, t_SPD on SPD
  uint32_t hcg_time = exposure_time;
  uint32_t lcg_time = hcg_time;
  uint32_t spd_time = std::min(std::max((uint32_t)exposure_time, (exposure_time_max + VS_TIME_MAX_OX03C10) / 3), exposure_time_max + VS_TIME_MAX_OX03C10);
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

int OX03C10::getSlaveAddress(int port) const {
  assert(port >= 0 && port <= 2);
  return (int[]){0x6C, 0x20, 0x6C}[port];
}

float OX03C10::getExposureScore(float desired_ev, int exp_t, int exp_g_idx, float exp_gain, int gain_idx) const {
  float score = std::abs(desired_ev - (exp_t * exp_gain));
  float m = exp_g_idx > analog_gain_rec_idx ? analog_gain_cost_high : analog_gain_cost_low;
  score += std::abs(exp_g_idx - (int)analog_gain_rec_idx) * m;
  score += ((1 - analog_gain_cost_delta) +
            analog_gain_cost_delta * (exp_g_idx - analog_gain_min_idx) / (analog_gain_max_idx - analog_gain_min_idx)) *
           std::abs(exp_g_idx - gain_idx) * 5.0;
  return score;
}
