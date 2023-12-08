#include <cassert>

#include "system/camerad/cameras/camera_common.h"
#include "system/camerad/cameras/camera_qcom2.h"
#include "system/camerad/sensors/sensor.h"

namespace {

const size_t AR0231_REGISTERS_HEIGHT = 2;
// TODO: this extra height is universal and doesn't apply per camera
const size_t AR0231_STATS_HEIGHT = 2 + 8;

const float DC_GAIN_AR0231 = 2.5;

const float DC_GAIN_ON_GREY_AR0231 = 0.2;
const float DC_GAIN_OFF_GREY_AR0231 = 0.3;

const int DC_GAIN_MIN_WEIGHT_AR0231 = 0;
const int DC_GAIN_MAX_WEIGHT_AR0231 = 1;

const float TARGET_GREY_FACTOR_AR0231 = 1.0;

const float sensor_analog_gains_AR0231[] = {
    1.0 / 8.0, 2.0 / 8.0, 2.0 / 7.0, 3.0 / 7.0,  // 0, 1, 2, 3
    3.0 / 6.0, 4.0 / 6.0, 4.0 / 5.0, 5.0 / 5.0,  // 4, 5, 6, 7
    5.0 / 4.0, 6.0 / 4.0, 6.0 / 3.0, 7.0 / 3.0,  // 8, 9, 10, 11
    7.0 / 2.0, 8.0 / 2.0, 8.0 / 1.0};            // 12, 13, 14, 15 = bypass

const int ANALOG_GAIN_MIN_IDX_AR0231 = 0x1;  // 0.25x
const int ANALOG_GAIN_REC_IDX_AR0231 = 0x6;  // 0.8x
const int ANALOG_GAIN_MAX_IDX_AR0231 = 0xD;  // 4.0x
const int ANALOG_GAIN_COST_DELTA_AR0231 = 0;
const float ANALOG_GAIN_COST_LOW_AR0231 = 0.1;
const float ANALOG_GAIN_COST_HIGH_AR0231 = 5.0;

const int EXPOSURE_TIME_MIN_AR0231 = 2;       // with HDR, fastest ss
const int EXPOSURE_TIME_MAX_AR0231 = 0x0855;  // with HDR, slowest ss, 40ms

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

std::map<uint16_t, uint16_t> ar0231_parse_registers(CameraState *c, uint8_t *data, std::initializer_list<uint16_t> addrs) {
  if (c->ar0231_register_lut.empty()) {
    c->ar0231_register_lut = ar0231_build_register_lut(c, data);
  }

  std::map<uint16_t, uint16_t> registers;
  for (uint16_t addr : addrs) {
    auto offset = c->ar0231_register_lut[addr];
    registers[addr] = ((uint16_t)data[offset.first] << 8) | data[offset.second];
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

CameraAR0231::CameraAR0231() {
  frame_width = FRAME_WIDTH;
  frame_height = FRAME_HEIGHT;
  frame_stride = FRAME_STRIDE;
  extra_height = AR0231_REGISTERS_HEIGHT + AR0231_STATS_HEIGHT;

  registers_offset = 0;
  frame_offset = AR0231_REGISTERS_HEIGHT;
  stats_offset = AR0231_REGISTERS_HEIGHT + FRAME_HEIGHT;

  dc_gain_factor = DC_GAIN_AR0231;
  dc_gain_min_weight = DC_GAIN_MIN_WEIGHT_AR0231;
  dc_gain_max_weight = DC_GAIN_MAX_WEIGHT_AR0231;
  dc_gain_on_grey = DC_GAIN_ON_GREY_AR0231;
  dc_gain_off_grey = DC_GAIN_OFF_GREY_AR0231;
  exposure_time_min = EXPOSURE_TIME_MIN_AR0231;
  exposure_time_max = EXPOSURE_TIME_MAX_AR0231;
  analog_gain_min_idx = ANALOG_GAIN_MIN_IDX_AR0231;
  analog_gain_rec_idx = ANALOG_GAIN_REC_IDX_AR0231;
  analog_gain_max_idx = ANALOG_GAIN_MAX_IDX_AR0231;
  analog_gain_cost_delta = ANALOG_GAIN_COST_DELTA_AR0231;
  analog_gain_cost_low = ANALOG_GAIN_COST_LOW_AR0231;
  analog_gain_cost_high = ANALOG_GAIN_COST_HIGH_AR0231;
  for (int i = 0; i <= analog_gain_max_idx; i++) {
    sensor_analog_gains[i] = sensor_analog_gains_AR0231[i];
  }
  min_ev = exposure_time_min * sensor_analog_gains[analog_gain_min_idx];
  max_ev = exposure_time_max * dc_gain_factor * sensor_analog_gains[analog_gain_max_idx];
  target_grey_factor = TARGET_GREY_FACTOR_AR0231;
}

void ar0231_process_registers(MultiCameraState *s, CameraState *c, cereal::FrameData::Builder &framed) {
  const uint8_t expected_preamble[] = {0x0a, 0xaa, 0x55, 0x20, 0xa5, 0x55};
  uint8_t *data = (uint8_t *)c->buf.cur_camera_buf->addr + c->ci->registers_offset;

  if (memcmp(data, expected_preamble, std::size(expected_preamble)) != 0) {
    LOGE("unexpected register data found");
    return;
  }

  auto registers = ar0231_parse_registers(c, data, {0x2000, 0x2002, 0x20b0, 0x20b2, 0x30c6, 0x30c8, 0x30ca, 0x30cc});

  uint32_t frame_id = ((uint32_t)registers[0x2000] << 16) | registers[0x2002];
  framed.setFrameIdSensor(frame_id);

  float temp_0 = ar0231_parse_temp_sensor(registers[0x30c6], registers[0x30c8], registers[0x20b0]);
  float temp_1 = ar0231_parse_temp_sensor(registers[0x30ca], registers[0x30cc], registers[0x20b2]);
  framed.setTemperaturesC({temp_0, temp_1});
}


std::vector<struct i2c_random_wr_payload> ar0231_get_exp_registers(const CameraInfo *ci, int exposure_time, int new_exp_g, bool dc_gain_enabled) {
  uint16_t analog_gain_reg = 0xFF00 | (new_exp_g << 4) | new_exp_g;
  return {
    {0x3366, analog_gain_reg},
    {0x3362, (uint16_t)(dc_gain_enabled ? 0x1 : 0x0)},
    {0x3012, (uint16_t)exposure_time},
  };
}
