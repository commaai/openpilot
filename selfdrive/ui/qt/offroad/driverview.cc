#include "selfdrive/ui/qt/offroad/driverview.h"

#include <QPainter>

#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/util.h"

const int FACE_IMG_SIZE = 130;

DriverViewWindow::DriverViewWindow(QWidget* parent) : QWidget(parent) {
  setAttribute(Qt::WA_OpaquePaintEvent);
  layout = new QStackedLayout(this);
  layout->setStackingMode(QStackedLayout::StackAll);

  cameraView = new CameraViewWidget("camerad", VISION_STREAM_RGB_FRONT, true, this);
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

  const int width = 4 * height() / 3;
  const QRect rect2 = {rect().center().x() - width / 2, rect().top(), width, rect().height()};
  const QRect valid_rect = {is_rhd ? rect2.right() - rect2.height() / 2 : rect2.left(), rect2.top(), rect2.height() / 2, rect2.height()};

  // blackout
  const QColor bg(0, 0, 0, 140);
  const QRect& blackout_rect = Hardware::TICI() ? rect() : rect2;
  p.fillRect(blackout_rect.adjusted(0, 0, valid_rect.left() - blackout_rect.right(), 0), bg);
  p.fillRect(blackout_rect.adjusted(valid_rect.right() - blackout_rect.left(), 0, 0, 0), bg);
  if (Hardware::TICI()) {
    p.fillRect(blackout_rect.adjusted(valid_rect.left()-blackout_rect.left()+1, 0, valid_rect.right()-blackout_rect.right()-1, -valid_rect.height()*7/10), bg); // top dz
  }

  // face bounding box
  cereal::DriverState::Reader driver_state = sm["driverState"].getDriverState();
  bool face_detected = driver_state.getFaceProb() > 0.5;
  if (face_detected) {
    auto fxy_list = driver_state.getFacePosition();
    auto std_list = driver_state.getFaceOrientationStd();
    float face_x = fxy_list[0];
    float face_y = fxy_list[1];
    float face_std = std::max(std_list[0], std_list[1]);

    float alpha = 0.7;
    if (face_std > 0.08) {
      alpha = std::max(0.7 - (face_std-0.08)*7, 0.0);
    }
    const int box_size = 0.6 * rect2.height() / 2;
    const float rhd_offset = 0.05; // lhd is shifted, so rhd is not mirrored
    int fbox_x = valid_rect.center().x() + (is_rhd ? (face_x + rhd_offset) : -face_x) * valid_rect.width();
    int fbox_y = valid_rect.center().y() + face_y * valid_rect.height();
    p.setPen(QPen(QColor(255, 255, 255, alpha * 255), 10));
    p.drawRoundedRect(fbox_x - box_size / 2, fbox_y - box_size / 2, box_size, box_size, 35.0, 35.0);
  }

  // icon
  const int img_offset = 30;
  const int img_x = is_rhd ? rect2.right() - FACE_IMG_SIZE - img_offset : rect2.left() + img_offset;
  const int img_y = rect2.bottom() - FACE_IMG_SIZE - img_offset;
  p.setOpacity(face_detected ? 1.0 : 0.3);
  p.drawPixmap(img_x, img_y, face_img);
}
