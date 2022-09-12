#include "system/camerad/cameras/camera.h"

#include <cassert>

#include "common/swaglog.h"
#include "common/util.h"

const float sensor_analog_gains_AR0231[] = {
  1.0/8.0, 2.0/8.0, 2.0/7.0, 3.0/7.0, // 0, 1, 2, 3
  3.0/6.0, 4.0/6.0, 4.0/5.0, 5.0/5.0, // 4, 5, 6, 7
  5.0/4.0, 6.0/4.0, 6.0/3.0, 7.0/3.0, // 8, 9, 10, 11
  7.0/2.0, 8.0/2.0, 8.0/1.0};         // 12, 13, 14, 15 = bypass

const size_t AR0231_REGISTERS_HEIGHT = 2;
// TODO: this extra height is universal and doesn't apply per camera
const size_t AR0231_STATS_HEIGHT = 2+8;

// AR0231
CameraAR0231::CameraAR0231() {
  id = CAMERA_ID_OX03C10;
  ci = {
    .frame_width = FRAME_WIDTH,
    .frame_height = FRAME_HEIGHT,
    .frame_stride = FRAME_STRIDE,
    .extra_height = AR0231_REGISTERS_HEIGHT + AR0231_STATS_HEIGHT,

    .registers_offset = 0,
    .frame_offset = AR0231_REGISTERS_HEIGHT,
    .stats_offset = AR0231_REGISTERS_HEIGHT + FRAME_HEIGHT,
  };

  dc_gain_factor = 2.5;
  dc_gain_max_weight = 1;
  dc_gain_on_grey = 0.2;
  dc_gain_off_grey = 0.3;
  exposure_time_min = 2; // with HDR, fastest ss
  exposure_time_max = 0x0855; // with HDR, slowest ss, 40ms
  analog_gain_min_idx = 0x1; // 0.25x
  analog_gain_rec_idx = 0x6; // 0.8x
  analog_gain_max_idx = 0xD; // 4.0x

  for (int i = 0; i <= analog_gain_max_idx; i++) {
    sensor_analog_gains[i] = sensor_analog_gains_AR0231[i];
  }
  min_ev = exposure_time_min * sensor_analog_gains[analog_gain_min_idx];
  max_ev = exposure_time_max * dc_gain_factor * sensor_analog_gains[analog_gain_max_idx];

  for (auto &v : start_reg_array_ar0231) {
    start_reg_array.push_back(v);
  }

  for (auto &v : init_array_ar0231) {
    init_array.push_back(v);
  }

  i2c_type = CAMERA_SENSOR_I2C_TYPE_WORD;
  in_port_info_dt = 0x12;  // Changing stats to 0x2C doesn't work, so change pixels to 0x12 instead
  reg_addr = 0x3000;
  expected_data = 0x354;
  config_val_low = 19200000;
}

CameraAR0231::~CameraAR0231() {

}

std::vector<struct i2c_random_wr_payload> CameraAR0231::getExposureVector(int new_g, bool dc_gain_enabled, int exposure_time, int dc_gain_weight) const {
  uint16_t analog_gain_reg = 0xFF00 | (new_g << 4) | new_g;
  return {
      {0x3366, analog_gain_reg},
      {0x3362, (uint16_t)(dc_gain_enabled ? 0x1 : 0x0)},
      {0x3012, (uint16_t)exposure_time},
  };
}

std::map<uint16_t, std::pair<int, int>> CameraAR0231::buildRegisterLut(uint8_t *data) {
  // This function builds a lookup table from register address, to a pair of indices in the
  // buffer where to read this address. The buffer contains padding bytes,
  // as well as markers to indicate the type of the next byte.
  //
  // 0xAA is used to indicate the MSB of the address, 0xA5 for the LSB of the address.
  // Every byte of data (MSB and LSB) is preceded by 0x5A. Specifying an address is optional
  // for contiguous ranges. See page 27-29 of the AR0231 Developer guide for more information.

  int max_i[] = {1828 / 2 * 3, 1500 / 2 * 3};
  auto get_next_idx = [](int cur_idx) {
    return (cur_idx % 3 == 1) ? cur_idx + 2 : cur_idx + 1; // Every third byte is padding
  };

  std::map<uint16_t, std::pair<int, int>> registers;
  for (int register_row = 0; register_row < 2; register_row++) {
    uint8_t *registers_raw = data + ci.frame_stride * register_row;
    assert(registers_raw[0] == 0x0a); // Start of line

    int value_tag_count = 0;
    int first_val_idx = 0;
    uint16_t cur_addr = 0;

    for (int i = 1; i <= max_i[register_row]; i = get_next_idx(get_next_idx(i))) {
      int val_idx = get_next_idx(i);

      uint8_t tag = registers_raw[i];
      uint16_t val = registers_raw[val_idx];

      if (tag == 0xAA) { // Register MSB tag
        cur_addr = val << 8;
      } else if (tag == 0xA5) { // Register LSB tag
        cur_addr |= val;
        cur_addr -= 2; // Next value tag will increment address again
      } else if (tag == 0x5A) { // Value tag

        // First tag
        if (value_tag_count % 2 == 0) {
          cur_addr += 2;
          first_val_idx = val_idx;
        } else {
          registers[cur_addr] = std::make_pair(first_val_idx + ci.frame_stride * register_row, val_idx + ci.frame_stride * register_row);
        }

        value_tag_count++;
      }
    }
  }
  return registers;
}


std::map<uint16_t, uint16_t> CameraAR0231::parseRegisters(uint8_t *data, std::initializer_list<uint16_t> addrs) {
  if (ar0231_register_lut.empty()) {
    ar0231_register_lut = buildRegisterLut(data);
  }

  std::map<uint16_t, uint16_t> registers;
  for (uint16_t addr : addrs) {
    auto offset = ar0231_register_lut[addr];
    registers[addr] = ((uint16_t)data[offset.first] << 8) | data[offset.second];
  }
  return registers;
}

static float parseTempSensor(uint16_t calib1, uint16_t calib2, uint16_t data_reg) {
  // See AR0231 Developer Guide - page 36
  float slope = (125.0 - 55.0) / ((float)calib1 - (float)calib2);
  float t0 = 55.0 - slope * (float)calib2;
  return t0 + slope * (float)data_reg;
}


void CameraAR0231::processRegisters(void *addr, cereal::FrameData::Builder &framed) {
  const uint8_t expected_preamble[] = {0x0a, 0xaa, 0x55, 0x20, 0xa5, 0x55};
  uint8_t *data = (uint8_t *)addr + registers_offset;

  if (memcmp(data, expected_preamble, std::size(expected_preamble)) != 0) {
    LOGE("unexpected register data found");
    return;
  }

  auto registers = parseRegisters(data, {0x2000, 0x2002, 0x20b0, 0x20b2, 0x30c6, 0x30c8, 0x30ca, 0x30cc});

  uint32_t frame_id = ((uint32_t)registers[0x2000] << 16) | registers[0x2002];
  framed.setFrameIdSensor(frame_id);

  float temp_0 = parseTempSensor(registers[0x30c6], registers[0x30c8], registers[0x20b0]);
  float temp_1 = parseTempSensor(registers[0x30ca], registers[0x30cc], registers[0x20b2]);
  framed.setTemperaturesC({temp_0, temp_1});
}
