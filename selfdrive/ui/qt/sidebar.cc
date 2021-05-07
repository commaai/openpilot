#include "common/util.h"
#include "sidebar.h"
#include "qt_window.h"
#include "selfdrive/hardware/hw.h"

void Sidebar::drawMetric(QPainter &p, const QString &label, const QString &val, QColor c, int y) {
  const QRect rect = {30, y, 240, val.isEmpty() ? (label.contains("\n") ? 124 : 100) : 148};

  p.setPen(Qt::NoPen);
  p.setBrush(QBrush(c));
  p.setClipRect(rect.x() + 6, rect.y(), 18, rect.height(), Qt::ClipOperation::ReplaceClip);
  p.drawRoundedRect(QRect(rect.x() + 6, rect.y() + 6, 100, rect.height() - 12), 25, 25);
  p.setClipping(false);

  p.setBrush(Qt::NoBrush);
  p.setPen(QColor(0xff, 0xff, 0xff, 0x55));
  p.drawRoundedRect(rect, 20, 20);

  p.setPen(QColor(0xff, 0xff, 0xff));
  if (val.isEmpty()) {
    const QFont vf = QFont("sans-bold", 24);
    p.setFont(vf);
    const QRect r = QRect(rect.x() + 35, rect.y() + (label.contains("\n") ? 40 : 50), rect.width() - 50, rect.height() - 50);
    p.drawText(r, Qt::AlignCenter, label);
  } else {
    const QFont vf = QFont("sans-bold", 40);
    p.setFont(vf);
    p.drawText(rect.x() + 50, rect.y() + 50, val);

    const QFont lf = QFont("sans-regular", 24);
    p.setFont(lf);
    p.drawText(rect.x() + 50, rect.y() + 50 + 66, label);
  }
}

Sidebar::Sidebar(QWidget *parent) : QFrame(parent) {
  home_img = QImage("../assets/images/button_home.png").scaled(180, 180, Qt::KeepAspectRatio, Qt::SmoothTransformation);
  settings_img = QImage("../assets/images/button_settings.png").scaled(settings_btn.width(), settings_btn.height(), Qt::KeepAspectRatio, Qt::SmoothTransformation);;

  setFixedWidth(300);
  setMinimumHeight(vwp_h);
  setStyleSheet("background-color: rgb(57, 57, 57);");
}

void Sidebar::mousePressEvent(QMouseEvent *event) {
  if (settings_btn.contains(event->pos())) {
    emit openSettings();
  }
}

void Sidebar::update(const UIState &s) {
  /*
  static std::map<NetStatus, std::pair<QString, QColor>> connectivity_map = {
      {NET_ERROR, {"CONNECT\nERROR", COLOR_DANGER}},
      {NET_CONNECTED, {"CONNECT\nONLINE", COLOR_GOOD}},
      {NET_DISCONNECTED, {"CONNECT\nOFFLINE", COLOR_WARNING}},
  };
  auto net_params = connectivity_map[s.scene.athenaStatus];
  connect->update(net_params.first, net_params.second);

  static std::map<cereal::DeviceState::ThermalStatus, QColor> temp_severity_map = {
      {cereal::DeviceState::ThermalStatus::GREEN, COLOR_GOOD},
      {cereal::DeviceState::ThermalStatus::YELLOW, COLOR_WARNING},
      {cereal::DeviceState::ThermalStatus::RED, COLOR_DANGER},
      {cereal::DeviceState::ThermalStatus::DANGER, COLOR_DANGER}};
  QString temp_val = QString("%1 °C").arg((int)s.scene.deviceState.getAmbientTempC());
  temp->update(temp_val, temp_severity_map[s.scene.deviceState.getThermalStatus()], "TEMP");

  QColor panda_color = COLOR_GOOD;
  QString panda_message = "VEHICLE\nONLINE";
  if (s.scene.pandaType == cereal::PandaState::PandaType::UNKNOWN) {
    panda_color = COLOR_DANGER;
    panda_message = "NO\nPANDA";
  }
  else if (Hardware::TICI() && s.scene.started) {
    panda_color = s.scene.gpsOK ? COLOR_GOOD : COLOR_WARNING;
    panda_message = QString("SAT CNT\n%1").arg(s.scene.satelliteCount);
  }
  panda->update(panda_message, panda_color);
  */

  repaint();
}

void Sidebar::paintEvent(QPaintEvent *event) {
  QPainter p(this);
  p.setPen(Qt::NoPen);
  p.setRenderHint(QPainter::Antialiasing);

  QFont font = QFont("opensans");
  font.setPixelSize(48);
  p.setFont(font);

  // draw settings button
  p.setOpacity(0.65);
  p.drawImage(settings_btn.x(), settings_btn.y(), settings_img);

  // draw home button
  p.setOpacity(1.0);
  p.drawImage(60, 1080 - 180 - 40, home_img);

  // network
  p.drawImage(58, 196, signal_imgs[strength]);
  p.setPen(QColor(0xff, 0xff, 0xff));
  p.drawText(50, 273, network_type[net_type]);

  // temperature
  drawMetric(p, "TEMP", "40°C", warning_color, 338);

  // panda
  drawMetric(p, "VEHICLE\nONLINE", "", good_color, 518);

  // connect
  drawMetric(p, "CONNECT\nOFFLINE", "", danger_color, 676);
}
