#include "selfdrive/ui/qt/offroad/driverview.h"

#include <algorithm>
#include <QPainter>

#include "selfdrive/ui/qt/util.h"

DriverViewWindow::DriverViewWindow(QWidget* parent) : CameraWidget("camerad", VISION_STREAM_DRIVER, parent) {
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

  driver_monitor.updateState(*uiState());
  driver_monitor.draw(p, rect());
}

mat4 DriverViewWindow::calcFrameMatrix() {
  const float driver_view_ratio = 2.0;
  const float yscale = stream_height * driver_view_ratio / stream_width;
  const float xscale = yscale * glHeight() / glWidth() * stream_width / stream_height;
  return mat4{{
    xscale,  0.0, 0.0, 0.0,
    0.0,  yscale, 0.0, 0.0,
    0.0,  0.0, 1.0, 0.0,
    0.0,  0.0, 0.0, 1.0,
  }};
}

DriverViewDialog::DriverViewDialog(QWidget *parent) : DialogBase(parent) {
  Params().putBool("IsDriverViewEnabled", true);
  device()->resetInteractiveTimeout(60);

  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(0, 0, 0, 0);
  auto camera = new DriverViewWindow(this);
  main_layout->addWidget(camera);
  QObject::connect(camera, &DriverViewWindow::clicked, this, &DialogBase::accept);
  QObject::connect(device(), &Device::interactiveTimeout, this, &DialogBase::accept);
}

void DriverViewDialog::done(int r) {
  Params().putBool("IsDriverViewEnabled", false);
  QDialog::done(r);
}
