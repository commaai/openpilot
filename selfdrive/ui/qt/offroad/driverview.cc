#include "selfdrive/ui/qt/offroad/driverview.h"

#include <algorithm>
#include <QVBoxLayout>
#include <QPainter>

#include "selfdrive/ui/qt/util.h"

DriverViewWindow::DriverViewWindow(QWidget* parent) : CameraWidget("camerad", VISION_STREAM_DRIVER, parent) {
  main_layout = new QVBoxLayout(this);
  main_layout->setMargin(UI_BORDER_SIZE);
  main_layout->setSpacing(0);

  driver_monitor = new DriverMonitorRenderer(this);
  main_layout->addWidget(driver_monitor, 0, Qt::AlignBottom | Qt::AlignLeft);

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
  driver_monitor->setVisible(true);
}

void DriverViewWindow::hideEvent(QHideEvent* event) {
  params.putBool("IsDriverViewEnabled", false);
  stopVipcThread();
  CameraWidget::hideEvent(event);
  driver_monitor->setVisible(false);
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
  bool new_is_rhd = driver_state.getWheelOnRightProb() > 0.5;
  if (is_rhd != new_is_rhd) {
    is_rhd = new_is_rhd;
    main_layout->removeWidget(driver_monitor);
    main_layout->addWidget(driver_monitor, 0, Qt::AlignBottom | (is_rhd ? Qt::AlignRight : Qt::AlignLeft));
  }
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
