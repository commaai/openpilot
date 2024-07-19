#pragma once

#include <mutex>
#include <vector>

#include "common/util.h"
#include "system/camerad/sensors/sensor.h"

class CameraExposure {
public:
  CameraExposure(int camera_num, const SensorInfo *sensor_info, int width, int height, float focal_len);
  std::vector<i2c_random_wr_payload> getExposureRegisters(const CameraBuf *b, int x_skip, int y_skip);
  float valuePercent(int index) const;
  void setFrameMetaData(FrameMetadata &meta_data);
  static float setExposureTarget(const CameraBuf *b, const Rect &ae_xywh, int x_skip, int y_skip);

protected:
  void updateScore(float desired_ev, int exp_t, int exp_g_idx, float exp_gain);

  int exposure_time;
  bool dc_gain_enabled;
  int dc_gain_weight;
  int gain_idx;
  float analog_gain_frac;

  float cur_ev[3];
  float best_ev_score;
  int new_exp_g;
  int new_exp_t;
  float fl_pix;

  Rect ae_xywh;
  float measured_grey_fraction;
  float target_grey_fraction;

  std::mutex exp_lock;
  const SensorInfo *ci = nullptr;
};
