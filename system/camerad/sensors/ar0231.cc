#include <cassert>

#include "common/swaglog.h"
#include "system/camerad/cameras/camera_common.h"
#include "system/camerad/cameras/camera_qcom2.h"
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

std::map<uint16_t, std::pair<int, int>> ar0231_build_register_lut(CameraState *c, uint8_t *data) {
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
    uint8_t *registers_raw = data + c->ci->frame_stride * register_row;
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
          registers[cur_addr] = std::make_pair(first_val_idx + c->ci->frame_stride * register_row, val_idx + c->ci->frame_stride * register_row);
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
  mipi_format = CAM_FORMAT_MIPI_RAW_12;
  frame_data_type = 0x12;  // Changing stats to 0x2C doesn't work, so change pixels to 0x12 instead
  mclk_frequency = 19200000; //Hz

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
}

void AR0231::processRegisters(CameraState *c, cereal::FrameData::Builder &framed) const {
  const uint8_t expected_preamble[] = {0x0a, 0xaa, 0x55, 0x20, 0xa5, 0x55};
  uint8_t *data = (uint8_t *)c->buf.cur_camera_buf->addr + c->ci->registers_offset;

  if (memcmp(data, expected_preamble, std::size(expected_preamble)) != 0) {
    LOGE("unexpected register data found");
    return;
  }

  if (ar0231_register_lut.empty()) {
    ar0231_register_lut = ar0231_build_register_lut(c, data);
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
