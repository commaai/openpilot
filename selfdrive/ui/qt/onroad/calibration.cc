#include "selfdrive/ui/qt/onroad/calibration.h"

const Eigen::Matrix3d VIEW_FROM_DEVICE = (Eigen::Matrix3d() <<
  0.0, 1.0, 0.0,
  0.0, 0.0, 1.0,
  1.0, 0.0, 0.0).finished();

const Eigen::Matrix3d FCAM_INTRINSIC_MATRIX = (Eigen::Matrix3d() <<
  2648.0, 0.0, 1928.0 / 2,
  0.0, 2648.0, 1208.0 / 2,
  0.0, 0.0, 1.0).finished();

// tici ecam focal probably wrong? magnification is not consistent across frame
// Need to retrain model before this can be changed
const Eigen::Matrix3d ECAM_INTRINSIC_MATRIX = (Eigen::Matrix3d() <<
  567.0, 0.0, 1928.0 / 2,
  0.0, 567.0, 1208.0 / 2,
  0.0, 0.0, 1.0).finished();

void Calibration::update(bool wide_cam, const cereal::LiveCalibrationData::Reader &live_calib, int width, int height) {
  auto _euler2rot = [](const capnp::List<float>::Reader &rpy_list) {
    return euler2rot({rpy_list[0], rpy_list[1], rpy_list[2]});
  };

  if (live_calib.getCalStatus() == cereal::LiveCalibrationData::Status::CALIBRATED) {
    view_from_calib = VIEW_FROM_DEVICE * _euler2rot(live_calib.getRpyCalib());
    if (wide_cam) {
      view_from_calib *= _euler2rot(live_calib.getWideFromDeviceEuler());
    }
  } else {
    view_from_calib = VIEW_FROM_DEVICE;
  }

  intrinsic_matrix = wide_cam ? ECAM_INTRINSIC_MATRIX: FCAM_INTRINSIC_MATRIX;
  updateCalibration(width, height, wide_cam ? 2.0 : 1.1);
}

void Calibration::updateCalibration(int width, int height, float zoom) {
  // Project point at "infinity" to compute x and y offsets
  // to ensure this ends up in the middle of the screen
  // for narrow come and a little lower for wide cam.
  // TODO: use proper perspective transform?

  Eigen::Vector3d inf(1000., 0., 0.);
  auto Ep = view_from_calib * inf;
  auto Kep = intrinsic_matrix * Ep;

  float center_x = intrinsic_matrix(0, 2);
  float center_y = intrinsic_matrix(1, 2);

  float max_x_offset = center_x * zoom - width / 2 - 5;
  float max_y_offset = center_y * zoom - height / 2 - 5;

  float x_offset = std::clamp<float>((Kep.x() / Kep.z() - center_x) * zoom, -max_x_offset, max_x_offset);
  float y_offset = std::clamp<float>((Kep.y() / Kep.z() - center_y) * zoom, -max_y_offset, max_y_offset);

  float zx = zoom * 2 * center_x / width;
  float zy = zoom * 2 * center_y / height;

  frame_matrix = mat4{{
    zx, 0.0, 0.0, -x_offset / width * 2,
    0.0, zy, 0.0, y_offset / height * 2,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
  }};

  // Apply transformation such that video pixel coordinates match video
  // 1) Put (0, 0) in the middle of the video
  // 2) Apply same scaling as video
  // 3) Put (0, 0) in top left corner of video
  car_space_transform.reset();
  car_space_transform.translate(width / 2 - x_offset, height / 2 - y_offset)
      .scale(zoom, zoom)
      .translate(-center_x, -center_y);

  clip_region = QRectF{-clip_margin, -clip_margin, width + 2 * clip_margin, height + 2 * clip_margin};
}

// Projects a point in car to space to the corresponding point in full frame image space.
bool Calibration::mapToFrame(float x, float y, float z, QPointF *out) const {
  Eigen::Vector3d pt(x, y, z);
  auto Ep = view_from_calib * pt;
  auto KEp = intrinsic_matrix * Ep;
  QPointF point = car_space_transform.map(QPointF{KEp.x() / KEp.z(), KEp.y() / KEp.z()});

  if (clip_region.contains(point)) {
    *out = point;
    return true;
  }
  return false;
}
