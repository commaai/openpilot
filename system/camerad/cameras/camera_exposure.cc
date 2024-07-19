#include "system/camerad/cameras/camera_exposure.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

#include "common/params.h"

CameraExposure::CameraExposure(int camera_num, const SensorInfo *sensor_info, int width, int height, float focal_len)
    : ci(sensor_info) {
  target_grey_fraction = 0.3;
  dc_gain_enabled = false;
  dc_gain_weight = ci->dc_gain_min_weight;
  gain_idx = ci->analog_gain_rec_idx;
  exposure_time = 5;
  cur_ev[0] = cur_ev[1] = cur_ev[2] = (1 + dc_gain_weight * (ci->dc_gain_factor - 1) / ci->dc_gain_max_weight) * ci->sensor_analog_gains[gain_idx] * exposure_time;

  fl_pix = focal_len / ci->pixel_size_mm;

  // set areas for each camera, shouldn't be changed
  std::vector<std::pair<Rect, float>> ae_targets = {
    // (Rect, F)
    std::make_pair((Rect){96, 250, 1734, 524}, 567.0),   // wide
    std::make_pair((Rect){96, 160, 1734, 986}, 2648.0),  // road
    std::make_pair((Rect){96, 242, 1736, 906}, 567.0)    // driver
  };
  int h_ref = 1208;
  /*
    exposure target intrinics is
    [
      [F, 0, 0.5*ae_xywh[2]]
      [0, F, 0.5*H-ae_xywh[1]]
      [0, 0, 1]
    ]
  */
  auto ae_target = ae_targets[camera_num];
  Rect xywh_ref = ae_target.first;
  float fl_ref = ae_target.second;

  ae_xywh = (Rect){
    std::max(0, width / 2 - (int)(fl_pix / fl_ref * xywh_ref.w / 2)),
    std::max(0, height / 2 - (int)(fl_pix / fl_ref * (h_ref / 2 - xywh_ref.y))),
    std::min((int)(fl_pix / fl_ref * xywh_ref.w), width / 2 + (int)(fl_pix / fl_ref * xywh_ref.w / 2)),
    std::min((int)(fl_pix / fl_ref * xywh_ref.h), height / 2 + (int)(fl_pix / fl_ref * (h_ref / 2 - xywh_ref.y)))};
}

std::vector<i2c_random_wr_payload> CameraExposure::getExposureRegisters(const CameraBuf *buf, int x_skip, int y_skip) {
  const float dt = 0.05;

  const float ts_grey = 10.0;
  const float ts_ev = 0.05;

  const float k_grey = (dt / ts_grey) / (1.0 + dt / ts_grey);
  const float k_ev = (dt / ts_ev) / (1.0 + dt / ts_ev);

  float grey_frac = setExposureTarget(buf, ae_xywh, x_skip, y_skip);
  // It takes 3 frames for the commanded exposure settings to take effect. The first frame is already started by the time
  // we reach this function, the other 2 are due to the register buffering in the sensor.
  // Therefore we use the target EV from 3 frames ago, the grey fraction that was just measured was the result of that control action.
  // TODO: Lower latency to 2 frames, by using the histogram outputted by the sensor we can do AE before the debayering is complete

  const float cur_ev_ = cur_ev[buf->cur_frame_data.frame_id % 3];

  // Scale target grey between 0.1 and 0.4 depending on lighting conditions
  float new_target_grey = std::clamp(0.4 - 0.3 * log2(1.0 + ci->target_grey_factor * cur_ev_) / log2(6000.0), 0.1, 0.4);
  float target_grey = (1.0 - k_grey) * target_grey_fraction + k_grey * new_target_grey;

  float desired_ev = std::clamp(cur_ev_ * target_grey / grey_frac, ci->min_ev, ci->max_ev);
  float k = (1.0 - k_ev) / 3.0;
  desired_ev = (k * cur_ev[0]) + (k * cur_ev[1]) + (k * cur_ev[2]) + (k_ev * desired_ev);

  best_ev_score = 1e6;
  new_exp_g = 0;
  new_exp_t = 0;

  // Hysteresis around high conversion gain
  // We usually want this on since it results in lower noise, but turn off in very bright day scenes
  bool enable_dc_gain = dc_gain_enabled;
  if (!enable_dc_gain && target_grey < ci->dc_gain_on_grey) {
    enable_dc_gain = true;
    dc_gain_weight = ci->dc_gain_min_weight;
  } else if (enable_dc_gain && target_grey > ci->dc_gain_off_grey) {
    enable_dc_gain = false;
    dc_gain_weight = ci->dc_gain_max_weight;
  }

  if (enable_dc_gain && dc_gain_weight < ci->dc_gain_max_weight) {
    dc_gain_weight += 1;
  }
  if (!enable_dc_gain && dc_gain_weight > ci->dc_gain_min_weight) {
    dc_gain_weight -= 1;
  }

  std::string gain_bytes, time_bytes;
  if (env_ctrl_exp_from_params) {
    Params params;
    gain_bytes = params.get("CameraDebugExpGain");
    time_bytes = params.get("CameraDebugExpTime");
  }

  if (gain_bytes.size() > 0 && time_bytes.size() > 0) {
    // Override gain and exposure time
    gain_idx = std::stoi(gain_bytes);
    exposure_time = std::stoi(time_bytes);

    new_exp_g = gain_idx;
    new_exp_t = exposure_time;
    enable_dc_gain = false;
  } else {
    // Simple brute force optimizer to choose sensor parameters
    // to reach desired EV
    for (int g = std::max((int)ci->analog_gain_min_idx, gain_idx - 1); g <= std::min((int)ci->analog_gain_max_idx, gain_idx + 1); g++) {
      float gain = ci->sensor_analog_gains[g] * (1 + dc_gain_weight * (ci->dc_gain_factor - 1) / ci->dc_gain_max_weight);

      // Compute optimal time for given gain
      int t = std::clamp(int(std::round(desired_ev / gain)), ci->exposure_time_min, ci->exposure_time_max);

      // Only go below recommended gain when absolutely necessary to not overexpose
      if (g < ci->analog_gain_rec_idx && t > 20 && g < gain_idx) {
        continue;
      }

      updateScore(desired_ev, t, g, gain);
    }
  }

  exp_lock.lock();

  measured_grey_fraction = grey_frac;
  target_grey_fraction = target_grey;

  analog_gain_frac = ci->sensor_analog_gains[new_exp_g];
  gain_idx = new_exp_g;
  exposure_time = new_exp_t;
  dc_gain_enabled = enable_dc_gain;

  float gain = analog_gain_frac * (1 + dc_gain_weight * (ci->dc_gain_factor - 1) / ci->dc_gain_max_weight);
  cur_ev[buf->cur_frame_data.frame_id % 3] = exposure_time * gain;

  exp_lock.unlock();

  // Processing a frame takes right about 50ms, so we need to wait a few ms
  // so we don't send i2c commands around the frame start.
  int ms = (nanos_since_boot() - buf->cur_frame_data.timestamp_sof) / 1000000;
  if (ms < 60) {
    util::sleep_for(60 - ms);
  }
  // LOGE("ae - camera %d, cur_t %.5f, sof %.5f, dt %.5f", camera_num, 1e-9 * nanos_since_boot(), 1e-9 * buf.cur_frame_data.timestamp_sof, 1e-9 * (nanos_since_boot() - buf.cur_frame_data.timestamp_sof));

  return ci->getExposureRegisters(exposure_time, new_exp_g, dc_gain_enabled);
}

void CameraExposure::setFrameMetaData(FrameMetadata &meta_data) {
  exp_lock.lock();
  meta_data.gain = analog_gain_frac * (1 + dc_gain_weight * (ci->dc_gain_factor - 1) / ci->dc_gain_max_weight);
  meta_data.high_conversion_gain = dc_gain_enabled;
  meta_data.integ_lines = exposure_time;
  meta_data.measured_grey_fraction = measured_grey_fraction;
  meta_data.target_grey_fraction = target_grey_fraction;
  exp_lock.unlock();
}

void CameraExposure::updateScore(float desired_ev, int exp_t, int exp_g_idx, float exp_gain) {
  float score = ci->getExposureScore(desired_ev, exp_t, exp_g_idx, exp_gain, gain_idx);
  if (score < best_ev_score) {
    new_exp_t = exp_t;
    new_exp_g = exp_g_idx;
    best_ev_score = score;
  }
}

float CameraExposure::setExposureTarget(const CameraBuf *b, const Rect &ae_xywh, int x_skip, int y_skip) {
  int lum_med;
  uint32_t lum_binning[256] = {0};
  const uint8_t *pix_ptr = b->cur_yuv_buf->y;

  unsigned int lum_total = 0;
  for (int y = ae_xywh.y; y < ae_xywh.y + ae_xywh.h; y += y_skip) {
    for (int x = ae_xywh.x; x < ae_xywh.x + ae_xywh.w; x += x_skip) {
      uint8_t lum = pix_ptr[(y * b->rgb_width) + x];
      lum_binning[lum]++;
      lum_total += 1;
    }
  }

  // Find mean lumimance value
  unsigned int lum_cur = 0;
  for (lum_med = 255; lum_med >= 0; lum_med--) {
    lum_cur += lum_binning[lum_med];

    if (lum_cur >= lum_total / 2) {
      break;
    }
  }

  return lum_med / 256.0;
}

float CameraExposure::valuePercent(int index) const {
  return util::map_val(cur_ev[index], ci->min_ev, ci->max_ev, 0.0f, 100.0f);
}
