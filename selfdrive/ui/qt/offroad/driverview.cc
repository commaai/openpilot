#include "selfdrive/ui/qt/offroad/driverview.h"

#include <QPainter>

#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/util.h"

const int FACE_IMG_SIZE = 130;

DriverViewWindow::DriverViewWindow(QWidget* parent) : QWidget(parent) {
  setAttribute(Qt::WA_OpaquePaintEvent);
  layout = new QStackedLayout(this);
  layout->setStackingMode(QStackedLayout::StackAll);

  cameraView = new CameraViewWidget(VISION_STREAM_RGB_FRONT, this);
  layout->addWidget(cameraView);

  scene = new DriverViewScene(this);
  connect(cameraView, &CameraViewWidget::frameUpdated, scene, &DriverViewScene::frameUpdated);
  layout->addWidget(scene);
  layout->setCurrentWidget(scene);
}

void DriverViewWindow::mousePressEvent(QMouseEvent* e) {
  emit done();
}

DriverViewScene::DriverViewScene(QWidget* parent) : sm({"driverState"}), QWidget(parent) {
  face = QImage("../assets/img_driver_face.png").scaled(FACE_IMG_SIZE, FACE_IMG_SIZE, Qt::KeepAspectRatio, Qt::SmoothTransformation);
}

void DriverViewScene::showEvent(QShowEvent* event) {
  frame_updated = false;
  is_rhd = params.getBool("IsRHD");
  params.putBool("IsDriverViewEnabled", true);
}

void DriverViewScene::hideEvent(QHideEvent* event) {
  params.putBool("IsDriverViewEnabled", false);
}

void DriverViewScene::frameUpdated() {
  frame_updated = true;
  sm.update(0);
  update();
}

void DriverViewScene::paintEvent(QPaintEvent* event) {
  QPainter p(this);

  // startup msg
  if (!frame_updated) {
    p.setPen(QColor(0xff, 0xff, 0xff));
    p.setRenderHint(QPainter::TextAntialiasing);
    configFont(p, "Inter", 100, "Bold");
    p.drawText(geometry(), Qt::AlignCenter, "camera starting");
    return;
  }

  const int width = 4 * height() / 3;
  const QRect rect2 = {rect().center().x() - width / 2, rect().top(), width, rect().height()};
  const QRect valid_rect = {is_rhd ? rect2.right() - rect2.height() / 2 : rect2.left(), rect2.top(), rect2.height() / 2, rect2.height()};

  // blackout
  const int blackout_x_r = valid_rect.right();
  const QRect& blackout_rect = Hardware::TICI() ? rect() : rect2;
  const int blackout_w_r = blackout_rect.right() - valid_rect.right();
  const int blackout_x_l = blackout_rect.left();
  const int blackout_w_l = valid_rect.left() - blackout_x_l;

  QColor bg(0, 0, 0, 140);
  p.setPen(QPen(bg));
  p.setBrush(QBrush(bg));
  p.drawRect(blackout_x_l, rect2.top(), blackout_w_l, rect2.height());
  p.drawRect(blackout_x_r, rect2.top(), blackout_w_r, rect2.height());

  // face bounding box
  cereal::DriverState::Reader driver_state = sm["driverState"].getDriverState();
  bool face_detected = driver_state.getFaceProb() > 0.4;
  if (face_detected) {
    auto fxy_list = driver_state.getFacePosition();
    float face_x = fxy_list[0];
    float face_y = fxy_list[1];
    int fbox_x = valid_rect.center().x() + (is_rhd ? face_x : -face_x) * valid_rect.width();
    int fbox_y = valid_rect.center().y() + face_y * valid_rect.height();

    float alpha = 0.2;
    face_x = std::abs(face_x);
    face_y = std::abs(face_y);
    if (face_x <= 0.35 && face_y <= 0.4) {
      alpha = 0.8 - (face_x > face_y ? face_x : face_y) * 0.6 / 0.375;
    }

    const int box_size = 0.6 * rect2.height() / 2;
    QPen pen(QColor(255, 255, 255, alpha * 255));
    pen.setWidth(10);
    p.setPen(pen);
    p.setBrush(Qt::NoBrush);
    p.drawRoundedRect(fbox_x - box_size / 2, fbox_y - box_size / 2, box_size, box_size, 35.0, 35.0);
  }

  // icon
  const int img_offset = 30;
  const int img_x = is_rhd ? rect2.right() - FACE_IMG_SIZE - img_offset : rect2.left() + img_offset;
  const int img_y = rect2.bottom() - FACE_IMG_SIZE - img_offset;
  p.setPen(Qt::NoPen);
  p.setOpacity(face_detected ? 1.0 : 0.3);
  p.drawImage(img_x, img_y, face);
}
