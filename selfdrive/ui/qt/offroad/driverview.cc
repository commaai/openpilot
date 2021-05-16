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

void DriverViewWindow::draw() {
  QPainter p(this);

  if (!connected()) {
    p.setPen(QColor(0xff, 0xff, 0xff));
    p.setRenderHint(QPainter::TextAntialiasing);
    configFont(p, "Open Sans", 100, "Bold");
    p.drawText(video_rect, Qt::AlignCenter, "Please wait for camera to start");
    return;
  }

  sm.update(0);

  const int width = 4 * video_rect.height() / 3;
  const QRect rect = {video_rect.center().x() - width / 2, video_rect.top(), width, video_rect.height()};  // x, y, w, h
  const QRect valid_rect = {is_rhd ? rect.right() - rect.height() / 2 : rect.left(), rect.top(), rect.height() / 2, rect.height()};

  // blackout
  const int blackout_x_r = valid_rect.right();
  const QRect& blackout_rect = Hardware::TICI() ? video_rect : rect;
  const int blackout_w_r = blackout_rect.right() - valid_rect.right();
  const int blackout_x_l = blackout_rect.left();
  const int blackout_w_l = valid_rect.left() - blackout_x_l;

  QColor bg;
  bg.setRgbF(0, 0, 0, 0.56);
  p.setBrush(QBrush(bg));
  p.drawRect(blackout_x_l, rect.top(), blackout_w_l, rect.height());
  p.drawRect(blackout_x_r, rect.top(), blackout_w_r, rect.height());
  p.setBrush(Qt::NoBrush);

  cereal::DriverState::Reader driver_state = sm["driverState"].getDriverState();
  const bool face_detected = driver_state.getFaceProb() > 0.4;
  if (face_detected) {
    auto fxy_list = driver_state.getFacePosition();
    float face_x = fxy_list[0];
    float face_y = fxy_list[1];
    int fbox_x = valid_rect.center().x() + (is_rhd ? face_x : -face_x) * valid_rect.width();
    int fbox_y = valid_rect.center().y() + face_y * valid_rect.height();

    float alpha = 0.2;
    if (face_x = std::abs(face_x), face_y = std::abs(face_y); face_x <= 0.35 && face_y <= 0.4)
      alpha = 0.8 - (face_x > face_y ? face_x : face_y) * 0.6 / 0.375;

    const int box_size = 0.6 * rect.height() / 2;
    QColor color;
    color.setRgbF(1.0, 1.0, 1.0, alpha);
    QPen pen = QPen(QColor(0xff, 0xff, 0xff, alpha));
    pen.setWidth(10);
    p.setPen(pen);
    p.drawRect(fbox_x - box_size / 2, fbox_y - box_size / 2, box_size, box_size);
    p.setPen(Qt::NoPen);
  }
  const int face_radius = 85;
  const int img_x = is_rhd ? rect.right() - face_radius * 2 - bdr_s * 2 : rect.left() + bdr_s * 2;
  const int img_y = rect.bottom() - face_radius * 2 - bdr_s * 2.5;
  p.drawImage(img_x, img_y, face_img);
}
