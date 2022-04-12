#include "selfdrive/ui/qt/offroad/driverview.h"

#include <QPainter>

#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/util.h"

const int FACE_IMG_SIZE = 130;

DriverViewWindow::DriverViewWindow(QWidget* parent) : QWidget(parent) {
  setAttribute(Qt::WA_OpaquePaintEvent);
  layout = new QStackedLayout(this);
  layout->setStackingMode(QStackedLayout::StackAll);

  cameraView = new CameraViewWidget("camerad", VISION_STREAM_RGB_DRIVER, true, this);
  layout->addWidget(cameraView);

  scene = new DriverViewScene(this);
  connect(cameraView, &CameraViewWidget::vipcThreadFrameReceived, scene, &DriverViewScene::frameUpdated);
  layout->addWidget(scene);
  layout->setCurrentWidget(scene);
}

void DriverViewWindow::mouseReleaseEvent(QMouseEvent* e) {
  emit done();
}

DriverViewScene::DriverViewScene(QWidget* parent) : sm({"driverState"}), QWidget(parent) {
  face_img = loadPixmap("../assets/img_driver_face.png", {FACE_IMG_SIZE, FACE_IMG_SIZE});
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
    p.setPen(Qt::white);
    p.setRenderHint(QPainter::TextAntialiasing);
    configFont(p, "Inter", 100, "Bold");
    p.drawText(geometry(), Qt::AlignCenter, "camera starting");
    return;
  }

  const int width = 2 * height();
  const QRect rect2 = {rect().center().x() - width / 2, rect().top(), width, rect().height()};

  cereal::DriverState::Reader driver_state = sm["driverState"].getDriverState();
  bool face_detected = driver_state.getFaceProb() > 0.5;

  // icon
  const int img_offset = 60;
  const int img_x = rect2.left() + img_offset;
  const int img_y = rect2.bottom() - FACE_IMG_SIZE - img_offset;
  p.setOpacity(face_detected ? 1.0 : 0.2);
  p.drawPixmap(img_x, img_y, face_img);
}
