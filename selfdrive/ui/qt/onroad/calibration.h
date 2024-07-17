#pragma once

#include <QTransform>

#include "common/mat.h"
#include "common/transformations/orientation.hpp"
#include "cereal/messaging/messaging.h"

class Calibration {
 public:
  Calibration() = default;
  void update(bool wide_cam, const cereal::LiveCalibrationData::Reader &live_calib, int width, int height);
  bool mapToFrame(float x, float y, float z, QPointF *out) const;
  mat4 frameMatrix() const { return frame_matrix; }

protected:
  void updateCalibration(int width, int height, float zoom);

  const float clip_margin = 500.0f;
  Eigen::Matrix3d view_from_calib;
  Eigen::Matrix3d intrinsic_matrix;
  QTransform car_space_transform;
  mat4 frame_matrix;
  QRectF clip_region;
};
