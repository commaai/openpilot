#include <cmath>

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

void OS04C10::ife_downscale_configure() {
  out_scale = 2;

  pixel_size_mm = 0.002;
  frame_width = 2688;
  frame_height = 1520;
  exposure_time_max = 2352;

  init_reg_array.insert(init_reg_array.end(), std::begin(ife_downscale_override_array_os04c10), std::end(ife_downscale_override_array_os04c10));
}

OS04C10::OS04C10() {
  image_sensor = cereal::FrameData::ImageSensor::OS04C10;
  bayer_pattern = CAM_ISP_PATTERN_BAYER_BGBGBG;
  pixel_size_mm = 0.004;
  data_word = false;

  // hdr_offset = 64 * 2 + 8; // stagger
  frame_width = 1344;
  frame_height = 760; //760 * 2 + hdr_offset;
  frame_stride = (frame_width * 12 / 8); // no alignment

  extra_height = 0;
  frame_offset = 0;

  start_reg_array.assign(std::begin(start_reg_array_os04c10), std::end(start_reg_array_os04c10));
  init_reg_array.assign(std::begin(init_array_os04c10), std::end(init_array_os04c10));
  probe_reg_addr = 0x300a;
  probe_expected_data = 0x5304;
  bits_per_pixel = 12;
  mipi_format = CAM_FORMAT_MIPI_RAW_12;
  frame_data_type = 0x2c;
  mclk_frequency = 24000000; // Hz

  // TODO: this was set from logs. actually calculate it out
  readout_time_ns = 11000000;

  ev_scale = 150.0;
  dc_gain_factor = 1;
  dc_gain_min_weight = 1;  // always on is fine
  dc_gain_max_weight = 1;
  dc_gain_on_grey = 0.9;
  dc_gain_off_grey = 1.0;
  exposure_time_min = 2;
  exposure_time_max = 1684;
  analog_gain_min_idx = 0x0;
  analog_gain_rec_idx = 0x0;  // 1x
  analog_gain_max_idx = 0x28;
  analog_gain_cost_delta = -1;
  analog_gain_cost_low = 0.4;
  analog_gain_cost_high = 6.4;
  for (int i = 0; i <= analog_gain_max_idx; i++) {
    sensor_analog_gains[i] = sensor_analog_gains_OS04C10[i];
  }
  min_ev = exposure_time_min * sensor_analog_gains[analog_gain_min_idx];
  max_ev = exposure_time_max * dc_gain_factor * sensor_analog_gains[analog_gain_max_idx];
  target_grey_factor = 0.01;

  black_level = 48;
  color_correct_matrix = {
    0x000000c2, 0x00000fe0, 0x00000fde,
    0x00000fa7, 0x000000d9, 0x00001000,
    0x00000fca, 0x00000fef, 0x000000c7,
  };
  for (int i = 0; i < 65; i++) {
    float fx = i / 64.0;
    gamma_lut_rgb.push_back((uint32_t)((10*fx)/(1+9*fx)*1023.0 + 0.5));
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
    0x01064832, 0x00da26d1, 0x00bb25d9, 0x00aac556, 0x00a06503, 0x009a64d3, 0x009744ba, 0x009744ba, 0x009a24d1, 0x00a00500, 0x00aa2551, 0x00ba45d2, 0x00d826c1, 0x01040820, 0x013729b9, 0x0171ab8d, 0x01b36d9b,
    0x00eee777, 0x00c2c616, 0x00ae2571, 0x009fe4ff, 0x0096e4b7, 0x0090e487, 0x008d446a, 0x008d2469, 0x0090a485, 0x009684b4, 0x009f64fb, 0x00ad456a, 0x00c1a60d, 0x00eca765, 0x011fc8fe, 0x015a4ad2, 0x019c0ce0,
    0x00dee6f7, 0x00b9c5ce, 0x00a5652b, 0x009964cb, 0x00904482, 0x00892449, 0x0085842c, 0x0085642b, 0x0088e447, 0x008fe47f, 0x0098e4c7, 0x00a4c526, 0x00b8a5c5, 0x00dc86e4, 0x010fc87e, 0x014a2a51, 0x018c0c60,
    0x00d626b1, 0x00b4e5a7, 0x00a1e50f, 0x0095e4af, 0x008c2461, 0x00850428, 0x0081640b, 0x0081440a, 0x0084a425, 0x008ba45d, 0x009564ab, 0x00a1450a, 0x00b3c59e, 0x00d3e69f, 0x01070838, 0x01418a0c, 0x01834c1a,
    0x00d4c6a6, 0x00b425a1, 0x00a1450a, 0x009544aa, 0x008b645b, 0x00844422, 0x0080a405, 0x0080a405, 0x00840420, 0x008b0458, 0x0094c4a6, 0x00a0a505, 0x00b30598, 0x00d26693, 0x0105a82d, 0x01402a01, 0x0181ec0f,
    0x00daa6d5, 0x00b765bb, 0x00a3c51e, 0x0097a4bd, 0x008e4472, 0x00872439, 0x0083841c, 0x0083641b, 0x0086e437, 0x008de46f, 0x009724b9, 0x00a30518, 0x00b665b3, 0x00d866c3, 0x010b885c, 0x01460a30, 0x0187ec3f,
    0x00e80740, 0x00bec5f6, 0x00aa6553, 0x009d24e9, 0x009404a0, 0x008d846c, 0x0089e44f, 0x0089e44f, 0x008d446a, 0x0093c49e, 0x009ca4e5, 0x00a9854c, 0x00bdc5ee, 0x00e5a72d, 0x0118c8c6, 0x01534a9a, 0x01952ca9,
    0x00fca7e5, 0x00d06683, 0x00b5c5ae, 0x00a5852c, 0x009c84e4, 0x009664b3, 0x0093649b, 0x0093449a, 0x009624b1, 0x009c24e1, 0x00a50528, 0x00b4e5a7, 0x00ce8674, 0x00fa47d2, 0x012d696b, 0x0167eb3f, 0x01a9cd4e,
    0x011888c4, 0x00ec6763, 0x00c7863c, 0x00b4e5a7, 0x00a8a545, 0x00a1c50e, 0x009ec4f6, 0x009ea4f5, 0x00a1a50d, 0x00a82541, 0x00b445a2, 0x00c5e62f, 0x00ea6753, 0x011648b2, 0x01496a4b, 0x0183ec1f, 0x01c5ae2d,
    0x013bc9de, 0x010fa87d, 0x00eac756, 0x00cd466a, 0x00bc25e1, 0x00b405a0, 0x00afc57e, 0x00afa57d, 0x00b3a59d, 0x00bbc5de, 0x00cc0660, 0x00e92749, 0x010da86d, 0x013989cc, 0x016cab65, 0x01a72d39, 0x01e8ef47,
    0x01666b33, 0x013a49d2, 0x011568ab, 0x00f7e7bf, 0x00e1c70e, 0x00d2e697, 0x00cb665b, 0x00cb2659, 0x00d26693, 0x00e0c706, 0x00f6a7b5, 0x0113c89e, 0x013849c2, 0x01642b21, 0x01974cba, 0x01d1ce8e, 0x0213909c,
    0x01986cc3, 0x016c2b61, 0x01476a3b, 0x0129e94f, 0x0113a89d, 0x0104c826, 0x00fd47ea, 0x00fd27e9, 0x01044822, 0x0112c896, 0x0128a945, 0x0145ca2e, 0x016a4b52, 0x01960cb0, 0x01c92e49, 0x0203b01d, 0x0245922c,
    0x01d1ae8d, 0x01a58d2c, 0x0180ac05, 0x01632b19, 0x014cea67, 0x013e29f1, 0x013689b4, 0x013669b3, 0x013d89ec, 0x014c0a60, 0x0161eb0f, 0x017f0bf8, 0x01a38d1c, 0x01cf4e7a, 0x02029014, 0x023d11e8, 0x027ed3f6,
  };
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
