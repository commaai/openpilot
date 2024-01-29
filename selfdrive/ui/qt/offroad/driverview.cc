#include "selfdrive/ui/qt/offroad/driverview.h"

#include <algorithm>
#include <QPainter>

#include "selfdrive/ui/qt/util.h"

const int FACE_IMG_SIZE = 130;

DriverViewWindow::DriverViewWindow(QWidget* parent) : CameraWidget("camerad", VISION_STREAM_DRIVER, true, parent) {
  face_img = loadPixmap("../assets/img_driver_face_static.png", {FACE_IMG_SIZE, FACE_IMG_SIZE});
  QObject::connect(this, &CameraWidget::clicked, this, &DriverViewWindow::done);
  QObject::connect(device(), &Device::interactiveTimeout, this, [this]() {
    if (isVisible()) {
      emit done();
    }
  });
}

void DriverViewWindow::showEvent(QShowEvent* event) {
  params.putBool("IsDriverViewEnabled", true);
  device()->resetInteractiveTimeout(60);
  CameraWidget::showEvent(event);
}

void DriverViewWindow::hideEvent(QHideEvent* event) {
  params.putBool("IsDriverViewEnabled", false);
  stopVipcThread();
  CameraWidget::hideEvent(event);
}

void DriverViewWindow::paintGL() {
  CameraWidget::paintGL();

  std::lock_guard lk(frame_lock);
  QPainter p(this);
  // startup msg
  if (frames.empty()) {
    p.setPen(Qt::white);
    p.setRenderHint(QPainter::TextAntialiasing);
    p.setFont(InterFont(100, QFont::Bold));
    p.drawText(geometry(), Qt::AlignCenter, tr("camera starting"));
    return;
  }

  const auto &sm = *(uiState()->sm);
  cereal::DriverStateV2::Reader driver_state = sm["driverStateV2"].getDriverStateV2();
  bool is_rhd = driver_state.getWheelOnRightProb() > 0.5;
  auto driver_data = is_rhd ? driver_state.getRightDriverData() : driver_state.getLeftDriverData();

  bool face_detected = driver_data.getFaceProb() > 0.7;
  if (face_detected) {
    auto fxy_list = driver_data.getFacePosition();
    auto std_list = driver_data.getFaceOrientationStd();
    float face_x = fxy_list[0];
    float face_y = fxy_list[1];
    float face_std = std::max(std_list[0], std_list[1]);

    float alpha = 0.7;
    if (face_std > 0.15) {
      alpha = std::max(0.7 - (face_std-0.15)*3.5, 0.0);
    }
    const int box_size = 220;
    // use approx instead of distort_points
    int fbox_x = 1080.0 - 1714.0 * face_x;
    int fbox_y = -135.0 + (504.0 + std::abs(face_x)*112.0) + (1205.0 - std::abs(face_x)*724.0) * face_y;
    p.setPen(QPen(QColor(255, 255, 255, alpha * 255), 10));
    p.drawRoundedRect(fbox_x - box_size / 2, fbox_y - box_size / 2, box_size, box_size, 35.0, 35.0);
  }

  // icon
  const int img_offset = 60;
  const int img_x = is_rhd ? rect().right() - FACE_IMG_SIZE - img_offset : rect().left() + img_offset;
  const int img_y = rect().bottom() - FACE_IMG_SIZE - img_offset;
  p.setOpacity(face_detected ? 1.0 : 0.2);
  p.drawPixmap(img_x, img_y, face_img);
}
