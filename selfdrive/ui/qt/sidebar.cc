#include "selfdrive/ui/qt/sidebar.h"

#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/common/util.h"
#include "selfdrive/hardware/hw.h"

void configFont(QPainter &p, QString family, int size, int weight) {
  QFont f(family);
  f.setPixelSize(size);
  f.setWeight(weight);
  p.setFont(f);
}

void Sidebar::drawMetric(QPainter &p, const QString &label, const QString &val, QColor c, int y) {
  const QRect rect = {30, y, 240, val.isEmpty() ? (label.contains("\n") ? 124 : 100) : 148};

  p.setPen(Qt::NoPen);
  p.setBrush(QBrush(c));
  p.setClipRect(rect.x() + 6, rect.y(), 18, rect.height(), Qt::ClipOperation::ReplaceClip);
  p.drawRoundedRect(QRect(rect.x() + 6, rect.y() + 6, 100, rect.height() - 12), 10, 10);
  p.setClipping(false);

  QPen pen = QPen(QColor(0xff, 0xff, 0xff, 0x55));
  pen.setWidth(2);
  p.setPen(pen);
  p.setBrush(Qt::NoBrush);
  p.drawRoundedRect(rect, 20, 20);

  p.setPen(QColor(0xff, 0xff, 0xff));
  if (val.isEmpty()) {
    configFont(p, "Open Sans", 35, 500);
    const QRect r = QRect(rect.x() + 35, rect.y(), rect.width() - 50, rect.height());
    p.drawText(r, Qt::AlignCenter, label);
  } else {
    configFont(p, "Open Sans", 58, 500);
    p.drawText(rect.x() + 50, rect.y() + 71, val);
    configFont(p, "Open Sans", 35, 400);
    p.drawText(rect.x() + 50, rect.y() + 50 + 77, label);
  }
}

Sidebar::Sidebar(QWidget *parent) : QFrame(parent) {
  home_img = QImage("../assets/images/button_home.png").scaled(180, 180, Qt::KeepAspectRatio, Qt::SmoothTransformation);
  settings_img = QImage("../assets/images/button_settings.png").scaled(settings_btn.width(), settings_btn.height(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation);;

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
  if (s.sm->frame % (6*UI_FREQ) == 0) {
    connect_str = "OFFLINE";
    connect_status = warning_color;
    auto last_ping = params.get<float>("LastAthenaPingTime");
    if (last_ping) {
      bool online = nanos_since_boot() - *last_ping < 70e9;
      connect_str = online ? "ONLINE" : "ERROR";
      connect_status = online ? good_color : danger_color;
    }
    repaint();
  }

  net_type = s.scene.deviceState.getNetworkType();
  strength = s.scene.deviceState.getNetworkStrength();

  temp_status = danger_color;
  auto ts = s.scene.deviceState.getThermalStatus();
  if (ts == cereal::DeviceState::ThermalStatus::GREEN) {
    temp_status = good_color;
  } else if (ts == cereal::DeviceState::ThermalStatus::YELLOW) {
    temp_status = warning_color;
  }
  temp_val = (int)s.scene.deviceState.getAmbientTempC();

  panda_str = "VEHICLE\nONLINE";
  panda_status = good_color;
  if (s.scene.pandaType == cereal::PandaState::PandaType::UNKNOWN) {
    panda_status = danger_color;
    panda_str = "NO\nPANDA";
  } else if (Hardware::TICI() && s.scene.started) {
    panda_str = QString("SAT CNT\n%1").arg(s.scene.satelliteCount);
    panda_status = s.scene.gpsOK ? good_color : warning_color;
  }

  if (s.sm->updated("deviceState") || s.sm->updated("pandaState")) {
    repaint();
  }
}

void Sidebar::paintEvent(QPaintEvent *event) {
  QPainter p(this);
  p.setPen(Qt::NoPen);
  p.setRenderHint(QPainter::Antialiasing);

  // static imgs
  p.setOpacity(0.65);
  p.drawImage(settings_btn.x(), settings_btn.y(), settings_img);
  p.setOpacity(1.0);
  p.drawImage(60, 1080 - 180 - 40, home_img);

  // network
  p.drawImage(58, 196, signal_imgs[strength]);
  configFont(p, "Open Sans", 35, 400);
  p.setPen(QColor(0xff, 0xff, 0xff));
  const QRect r = QRect(50, 247, 100, 50);
  p.drawText(r, Qt::AlignCenter, network_type[net_type]);

  // metrics
  drawMetric(p, "TEMP", QString("%1°C").arg(temp_val), temp_status, 338);
  drawMetric(p, panda_str, "", panda_status, 518);
  drawMetric(p, "CONNECT\n" + connect_str, "", connect_status, 676);
}
