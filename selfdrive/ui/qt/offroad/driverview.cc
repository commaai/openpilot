#include "selfdrive/ui/qt/offroad/driverview.h"

#include <QPainter>

#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/util.h"

DriverViewWindow::DriverViewWindow(QWidget* parent) : sm({"driverState"}), CameraViewWidget(VISION_STREAM_RGB_FRONT, parent) {
  is_rhd = Params().getBool("IsRHD");
  face_img = QImage("../assets/img_driver_face").scaled(180, 180, Qt::KeepAspectRatio, Qt::SmoothTransformation);
}

void DriverViewWindow::showEvent(QShowEvent* event) {
  Params().putBool("IsDriverViewEnabled", true);
}

void DriverViewWindow::hideEvent(QHideEvent* event) {
  Params().putBool("IsDriverViewEnabled", false);
}

void DriverViewWindow::paintGL() {
  const NVGcolor color = bg_colors[STATUS_DISENGAGED];
  glClearColor(color.r, color.g, color.b, 1.0);
  glClear(GL_STENCIL_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

  const Rect viz_rect = Rect{bdr_s, bdr_s, vwp_w - 2 * bdr_s, vwp_h - 2 * bdr_s};
  glViewport(viz_rect.x, viz_rect.y, viz_rect.w, viz_rect.h);

  QPainter p(this);

  if (!connected()) {
    p.setPen(QColor(0xff, 0xff, 0xff));
    p.setRenderHint(QPainter::TextAntialiasing);
    configFont(p, "Open Sans", 100, "Bold");
    p.drawText(QRect{viz_rect.x, viz_rect.y, viz_rect.w, viz_rect.h}, Qt::AlignCenter, "Please wait for camera to start");
    return;
  }

  CameraViewWidget::paintGL();
  sm.update(0);

  const int width = 4 * viz_rect.h / 3;
  const Rect rect = {viz_rect.centerX() - width / 2, viz_rect.y, width, viz_rect.h};  // x, y, w, h
  const Rect valid_rect = {is_rhd ? rect.right() - rect.h / 2 : rect.x, rect.y, rect.h / 2, rect.h};

  // blackout
  const int blackout_x_r = valid_rect.right();
  const Rect& blackout_rect = Hardware::TICI() ? viz_rect : rect;
  const int blackout_w_r = blackout_rect.right() - valid_rect.right();
  const int blackout_x_l = blackout_rect.x;
  const int blackout_w_l = valid_rect.x - blackout_x_l;

  QColor bg;
  bg.setRgbF(0, 0, 0, 0.56);
  p.setBrush(QBrush(bg));
  p.drawRect(blackout_x_l, rect.y, blackout_w_l, rect.h);
  p.drawRect(blackout_x_r, rect.y, blackout_w_r, rect.h);
  p.setBrush(Qt::NoBrush);

  cereal::DriverState::Reader driver_state = sm["driverState"].getDriverState();
  const bool face_detected = driver_state.getFaceProb() > 0.4;
  if (face_detected) {
    auto fxy_list = driver_state.getFacePosition();
    float face_x = fxy_list[0];
    float face_y = fxy_list[1];
    int fbox_x = valid_rect.centerX() + (is_rhd ? face_x : -face_x) * valid_rect.w;
    int fbox_y = valid_rect.centerY() + face_y * valid_rect.h;

    float alpha = 0.2;
    if (face_x = std::abs(face_x), face_y = std::abs(face_y); face_x <= 0.35 && face_y <= 0.4)
      alpha = 0.8 - (face_x > face_y ? face_x : face_y) * 0.6 / 0.375;

    const int box_size = 0.6 * rect.h / 2;
    QColor color;
    color.setRgbF(1.0, 1.0, 1.0, alpha);
    QPen pen = QPen(QColor(0xff, 0xff, 0xff, alpha));
    pen.setWidth(10);
    p.setPen(pen);
    p.drawRect(fbox_x - box_size / 2, fbox_y - box_size / 2, box_size, box_size);
    p.setPen(Qt::NoPen);
  }
  const int face_radius = 85;
  const int img_x = is_rhd ? rect.right() - face_radius * 2 - bdr_s * 2 : rect.x + bdr_s * 2;
  const int img_y = rect.bottom() - face_radius * 2 - bdr_s * 2.5;
  p.drawImage(img_x, img_y, face_img);
}
