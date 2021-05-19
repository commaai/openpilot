#include "selfdrive/ui/qt/offroad/driverview.h"

#include <QPainter>

#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/util.h"

const int FACE_RADIUS = 85;
// class DriverViewWindow

DriverViewWindow::DriverViewWindow(QWidget* parent) : QWidget(parent) {
  setAttribute(Qt::WA_OpaquePaintEvent);
  layout = new QStackedLayout(this);
  layout->setStackingMode(QStackedLayout::StackAll);

  cameraView = new CameraViewWidget(VISION_STREAM_RGB_BACK, this);
  layout->addWidget(cameraView);

  scene = new DriverViewScene(this);
  connect(this, &DriverViewWindow::update, scene, &DriverViewScene::update);
  connect(cameraView, &CameraViewWidget::frameUpdated, [=] {
    if (!scene->frame_updated) {
      scene->frame_updated = true;
      scene->repaint();
    }
  });
  layout->addWidget(scene);
  layout->setCurrentWidget(scene);
}

void DriverViewWindow::showEvent(QShowEvent* event) {
  scene->frame_updated = false;
  scene->is_rhd = params.getBool("IsRHD");
  params.putBool("IsDriverViewEnabled", true);
}

void DriverViewWindow::hideEvent(QHideEvent* event) {
  params.putBool("IsDriverViewEnabled", false);
}

// class DirverViewElem

DriverViewScene::DriverViewScene(QWidget* parent) : QWidget(parent) {
  face_img = QImage("../assets/img_driver_face").scaled(FACE_RADIUS, FACE_RADIUS, Qt::KeepAspectRatio, Qt::SmoothTransformation);
}

void DriverViewScene::update(const UIState& s) {
  auto& sm = *(s.sm);
  if (sm.updated("driverState")) {
    cereal::DriverState::Reader driver_state = sm["driverState"].getDriverState();
    face_detected = driver_state.getFaceProb() > 0.4;
    auto fxy_list = driver_state.getFacePosition();
    face_x = fxy_list[0];
    face_y = fxy_list[1];
    QWidget::update();
  }
}

void DriverViewScene::paintEvent(QPaintEvent* event) {
  QPainter p(this);
  if (!frame_updated) {
    p.setPen(QColor(0xff, 0xff, 0xff));
    p.setRenderHint(QPainter::TextAntialiasing);
    configFont(p, "Open Sans", 100, "Bold");
    p.drawText(geometry(), Qt::AlignCenter, "Please wait for camera to start");
    return;
  }

  QRect video_rect = {bdr_s, bdr_s, vwp_w - 2 * bdr_s, vwp_h - 2 * bdr_s};
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
  bg.setRgbF(0.0, 0, 0, 0.56);
  p.setPen(QPen(bg));
  p.setBrush(QBrush(bg));
  p.drawRect(blackout_x_l, rect.top(), blackout_w_l, rect.height());
  p.drawRect(blackout_x_r, rect.top(), blackout_w_r, rect.height());
  p.setBrush(Qt::NoBrush);

  if (face_detected) {
    int fbox_x = valid_rect.center().x() + (is_rhd ? face_x : -face_x) * valid_rect.width();
    int fbox_y = valid_rect.center().y() + face_y * valid_rect.height();
    float alpha = 0.2;
    if (face_x = std::abs(face_x), face_y = std::abs(face_y); face_x <= 0.35 && face_y <= 0.4) {
      alpha = 0.8 - (face_x > face_y ? face_x : face_y) * 0.6 / 0.375;
    }

    const int box_size = 0.6 * rect.height() / 2;
    QColor color;
    color.setRgbF(1.0, 1.0, 1.0, alpha);
    QPen pen = QPen(color);
    pen.setWidth(10);
    p.setPen(pen);
    p.drawRoundedRect(fbox_x - box_size / 2, fbox_y - box_size / 2, box_size, box_size, 35.0, 35.0);
    p.setPen(Qt::NoPen);
  }
  const int img_x = is_rhd ? rect.right() - FACE_RADIUS * 2 - bdr_s * 2 : rect.left() + bdr_s * 2;
  const int img_y = rect.bottom() - FACE_RADIUS * 2 - bdr_s * 2.5;
  p.setOpacity(face_detected ? 1.0 : 0.15);
  p.drawImage(img_x, img_y, face_img);
}
