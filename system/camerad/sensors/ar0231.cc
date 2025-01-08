#include <cassert>
#include <cmath>

#include "common/swaglog.h"
#include "system/camerad/sensors/sensor.h"

namespace {

const size_t AR0231_REGISTERS_HEIGHT = 2;
// TODO: this extra height is universal and doesn't apply per camera
const size_t AR0231_STATS_HEIGHT = 2 + 8;

const float sensor_analog_gains_AR0231[] = {
    1.0 / 8.0, 2.0 / 8.0, 2.0 / 7.0, 3.0 / 7.0,  // 0, 1, 2, 3
    3.0 / 6.0, 4.0 / 6.0, 4.0 / 5.0, 5.0 / 5.0,  // 4, 5, 6, 7
    5.0 / 4.0, 6.0 / 4.0, 6.0 / 3.0, 7.0 / 3.0,  // 8, 9, 10, 11
    7.0 / 2.0, 8.0 / 2.0, 8.0 / 1.0};            // 12, 13, 14, 15 = bypass

std::map<uint16_t, std::pair<int, int>> ar0231_build_register_lut(const AR0231 *s, uint8_t *data) {
  // This function builds a lookup table from register address, to a pair of indices in the
  // buffer where to read this address. The buffer contains padding bytes,
  // as well as markers to indicate the type of the next byte.
  //
  // 0xAA is used to indicate the MSB of the address, 0xA5 for the LSB of the address.
  // Every byte of data (MSB and LSB) is preceded by 0x5A. Specifying an address is optional
  // for contiguous ranges. See page 27-29 of the AR0231 Developer guide for more information.

  int max_i[] = {1828 / 2 * 3, 1500 / 2 * 3};
  auto get_next_idx = [](int cur_idx) {
    return (cur_idx % 3 == 1) ? cur_idx + 2 : cur_idx + 1;  // Every third byte is padding
  };

  std::map<uint16_t, std::pair<int, int>> registers;
  for (int register_row = 0; register_row < 2; register_row++) {
    uint8_t *registers_raw = data + s->frame_stride * register_row;
    assert(registers_raw[0] == 0x0a);  // Start of line

    int value_tag_count = 0;
    int first_val_idx = 0;
    uint16_t cur_addr = 0;

    for (int i = 1; i <= max_i[register_row]; i = get_next_idx(get_next_idx(i))) {
      int val_idx = get_next_idx(i);

      uint8_t tag = registers_raw[i];
      uint16_t val = registers_raw[val_idx];

      if (tag == 0xAA) {  // Register MSB tag
        cur_addr = val << 8;
      } else if (tag == 0xA5) {  // Register LSB tag
        cur_addr |= val;
        cur_addr -= 2;           // Next value tag will increment address again
      } else if (tag == 0x5A) {  // Value tag

        // First tag
        if (value_tag_count % 2 == 0) {
          cur_addr += 2;
          first_val_idx = val_idx;
        } else {
          registers[cur_addr] = std::make_pair(first_val_idx + s->frame_stride * register_row, val_idx + s->frame_stride * register_row);
        }

        value_tag_count++;
      }
    }
  }
  return registers;
}

float ar0231_parse_temp_sensor(uint16_t calib1, uint16_t calib2, uint16_t data_reg) {
  // See AR0231 Developer Guide - page 36
  float slope = (125.0 - 55.0) / ((float)calib1 - (float)calib2);
  float t0 = 55.0 - slope * (float)calib2;
  return t0 + slope * (float)data_reg;
}

}  // namespace

AR0231::AR0231() {
  image_sensor = cereal::FrameData::ImageSensor::AR0231;
  bayer_pattern = CAM_ISP_PATTERN_BAYER_GRGRGR;
  pixel_size_mm = 0.003;
  data_word = true;
  frame_width = 1928;
  frame_height = 1208;
  frame_stride = (frame_width * 12 / 8) + 4;
  extra_height = AR0231_REGISTERS_HEIGHT + AR0231_STATS_HEIGHT;

  registers_offset = 0;
  frame_offset = AR0231_REGISTERS_HEIGHT;
  stats_offset = AR0231_REGISTERS_HEIGHT + frame_height;

  start_reg_array.assign(std::begin(start_reg_array_ar0231), std::end(start_reg_array_ar0231));
  init_reg_array.assign(std::begin(init_array_ar0231), std::end(init_array_ar0231));
  probe_reg_addr = 0x3000;
  probe_expected_data = 0x354;
  bits_per_pixel = 12;
  mipi_format = CAM_FORMAT_MIPI_RAW_12;
  frame_data_type = 0x12;  // Changing stats to 0x2C doesn't work, so change pixels to 0x12 instead
  mclk_frequency = 19200000; //Hz

  readout_time_ns = 22850000;

  dc_gain_factor = 2.5;
  dc_gain_min_weight = 0;
  dc_gain_max_weight = 1;
  dc_gain_on_grey = 0.2;
  dc_gain_off_grey = 0.3;
  exposure_time_min = 2;       // with HDR, fastest ss
  exposure_time_max = 0x0855;  // with HDR, slowest ss, 40ms
  analog_gain_min_idx = 0x1;   // 0.25x
  analog_gain_rec_idx = 0x6;   // 0.8x
  analog_gain_max_idx = 0xD;   // 4.0x
  analog_gain_cost_delta = 0;
  analog_gain_cost_low = 0.1;
  analog_gain_cost_high = 5.0;
  for (int i = 0; i <= analog_gain_max_idx; i++) {
    sensor_analog_gains[i] = sensor_analog_gains_AR0231[i];
  }
  min_ev = exposure_time_min * sensor_analog_gains[analog_gain_min_idx];
  max_ev = exposure_time_max * dc_gain_factor * sensor_analog_gains[analog_gain_max_idx];
  target_grey_factor = 1.0;

  black_level = 168;
  color_correct_matrix = {
    0x000000af, 0x00000ff9, 0x00000fd8,
    0x00000fbc, 0x000000bb, 0x00000009,
    0x00000fb6, 0x00000fe0, 0x000000ea,
  };
  for (int i = 0; i < 65; i++) {
    float fx = i / 64.0;
    const float gamma_k = 0.75;
    const float gamma_b = 0.125;
    const float mp = 0.01; // ideally midpoint should be adaptive
    const float rk = 9 - 100*mp;
    // poly approximation for s curve
    fx = (fx > mp) ?
      ((rk * (fx-mp) * (1-(gamma_k*mp+gamma_b)) * (1+1/(rk*(1-mp))) / (1+rk*(fx-mp))) + gamma_k*mp + gamma_b) :
      ((rk * (fx-mp) * (gamma_k*mp+gamma_b) * (1+1/(rk*mp)) / (1-rk*(fx-mp))) + gamma_k*mp + gamma_b);
    gamma_lut_rgb.push_back((uint32_t)(fx*1023.0 + 0.5));
  }
  prepare_gamma_lut();
  linearization_lut = {
    0x02000000, 0x02000000, 0x02000000, 0x02000000,
    0x020007ff, 0x020007ff, 0x020007ff, 0x020007ff,
    0x02000bff, 0x02000bff, 0x02000bff, 0x02000bff,
    0x020017ff, 0x020017ff, 0x020017ff, 0x020017ff,
    0x02001bff, 0x02001bff, 0x02001bff, 0x02001bff,
    0x020023ff, 0x020023ff, 0x020023ff, 0x020023ff,
    0x00003fff, 0x00003fff, 0x00003fff, 0x00003fff,
    0x00003fff, 0x00003fff, 0x00003fff, 0x00003fff,
    0x00003fff, 0x00003fff, 0x00003fff, 0x00003fff,
  };
  linearization_pts = {0x07ff0bff, 0x17ff1bff, 0x23ff3fff, 0x3fff3fff};
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

void AR0231::processRegisters(uint8_t *cur_buf, cereal::FrameData::Builder &framed) const {
  const uint8_t expected_preamble[] = {0x0a, 0xaa, 0x55, 0x20, 0xa5, 0x55};
  uint8_t *data = cur_buf + registers_offset;

  if (memcmp(data, expected_preamble, std::size(expected_preamble)) != 0) {
    LOGE("unexpected register data found");
    return;
  }

  if (ar0231_register_lut.empty()) {
    ar0231_register_lut = ar0231_build_register_lut(this, data);
  }
  std::map<uint16_t, uint16_t> registers;
  for (uint16_t addr : {0x2000, 0x2002, 0x20b0, 0x20b2, 0x30c6, 0x30c8, 0x30ca, 0x30cc}) {
    auto offset = ar0231_register_lut[addr];
    registers[addr] = ((uint16_t)data[offset.first] << 8) | data[offset.second];
  }

  uint32_t frame_id = ((uint32_t)registers[0x2000] << 16) | registers[0x2002];
  framed.setFrameIdSensor(frame_id);

  float temp_0 = ar0231_parse_temp_sensor(registers[0x30c6], registers[0x30c8], registers[0x20b0]);
  float temp_1 = ar0231_parse_temp_sensor(registers[0x30ca], registers[0x30cc], registers[0x20b2]);
  framed.setTemperaturesC({temp_0, temp_1});
}


std::vector<i2c_random_wr_payload> AR0231::getExposureRegisters(int exposure_time, int new_exp_g, bool dc_gain_enabled) const {
  uint16_t analog_gain_reg = 0xFF00 | (new_exp_g << 4) | new_exp_g;
  return {
    {0x3366, analog_gain_reg},
    {0x3362, (uint16_t)(dc_gain_enabled ? 0x1 : 0x0)},
    {0x3012, (uint16_t)exposure_time},
  };
}

int AR0231::getSlaveAddress(int port) const {
  assert(port >= 0 && port <= 2);
  return (int[]){0x20, 0x30, 0x20}[port];
}

float AR0231::getExposureScore(float desired_ev, int exp_t, int exp_g_idx, float exp_gain, int gain_idx) const {
  // Cost of ev diff
  float score = std::abs(desired_ev - (exp_t * exp_gain)) * 10;
  // Cost of absolute gain
  float m = exp_g_idx > analog_gain_rec_idx ? analog_gain_cost_high : analog_gain_cost_low;
  score += std::abs(exp_g_idx - (int)analog_gain_rec_idx) * m;
  // Cost of changing gain
  score += std::abs(exp_g_idx - gain_idx) * (score + 1.0) / 10.0;
  return score;
}
