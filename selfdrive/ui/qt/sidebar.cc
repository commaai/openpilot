#include "selfdrive/ui/qt/sidebar.h"

#include <QMouseEvent>

#include "selfdrive/ui/qt/util.h"

void Sidebar::drawMetric(QPainter &p, const QString &label, QColor c, int y) {
  const QRect rect = {30, y, 240, label.contains("\n") ? 124 : 100};

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
  configFont(p, "Open Sans", 35, "Bold");
  const QRect r = QRect(rect.x() + 30, rect.y(), rect.width() - 40, rect.height());
  p.drawText(r, Qt::AlignCenter, label);
}

Sidebar::Sidebar(QWidget *parent) : QFrame(parent) {
  home_img = loadPixmap("../assets/images/button_home.png", {180, 180});
  settings_img = loadPixmap("../assets/images/button_settings.png", settings_btn.size(), Qt::IgnoreAspectRatio);

  connect(this, &Sidebar::valueChanged, [=] { update(); });

  setAttribute(Qt::WA_OpaquePaintEvent);
  setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
  setFixedWidth(300);
}

void Sidebar::mouseReleaseEvent(QMouseEvent *event) {
  if (settings_btn.contains(event->pos())) {
    emit openSettings();
  }
}

void Sidebar::updateState(const UIState &s) {
  if (!isVisible()) return;

  auto &sm = *(s.sm);

  auto deviceState = sm["deviceState"].getDeviceState();
  setProperty("netType", network_type[deviceState.getNetworkType()]);
  int strength = (int)deviceState.getNetworkStrength();
  setProperty("netStrength", strength > 0 ? strength + 1 : 0);

  ItemStatus connectStatus;
  auto last_ping = deviceState.getLastAthenaPingTime();
  if (last_ping == 0) {
    connectStatus = params.getBool("PrimeRedirected") ? ItemStatus{"NO\nPRIME", danger_color} : ItemStatus{"CONNECT\nOFFLINE", warning_color};
  } else {
    connectStatus = nanos_since_boot() - last_ping < 80e9 ? ItemStatus{"CONNECT\nONLINE", good_color} : ItemStatus{"CONNECT\nERROR", danger_color};
  }
  setProperty("connectStatus", QVariant::fromValue(connectStatus));

  ItemStatus tempStatus = {"TEMP\nHIGH", danger_color};
  auto ts = deviceState.getThermalStatus();
  if (ts == cereal::DeviceState::ThermalStatus::GREEN) {
    tempStatus = {"TEMP\nGOOD", good_color};
  } else if (ts == cereal::DeviceState::ThermalStatus::YELLOW) {
    tempStatus = {"TEMP\nOK", warning_color};
  }
  setProperty("tempStatus", QVariant::fromValue(tempStatus));

  ItemStatus pandaStatus = {"VEHICLE\nONLINE", good_color};
  if (s.scene.pandaType == cereal::PandaState::PandaType::UNKNOWN) {
    pandaStatus = {"NO\nPANDA", danger_color};
  } else if (s.scene.started && !sm["liveLocationKalman"].getLiveLocationKalman().getGpsOK()) {
    pandaStatus = {"GPS\nSEARCHING", warning_color};
  }
  setProperty("pandaStatus", QVariant::fromValue(pandaStatus));
}

void Sidebar::paintEvent(QPaintEvent *event) {
  QPainter p(this);
  p.setPen(Qt::NoPen);
  p.setRenderHint(QPainter::Antialiasing);

  p.fillRect(rect(), QColor(57, 57, 57));

  // static imgs
  p.setOpacity(0.65);
  p.drawPixmap(settings_btn.x(), settings_btn.y(), settings_img);
  p.setOpacity(1.0);
  p.drawPixmap(60, 1080 - 180 - 40, home_img);

  // network
  int x = 58;
  const QColor gray(0x54, 0x54, 0x54);
  for (int i = 0; i < 5; ++i) {
    p.setBrush(i < net_strength ? Qt::white : gray);
    p.drawEllipse(x, 196, 27, 27);
    x += 37;
  }

  configFont(p, "Open Sans", 35, "Regular");
  p.setPen(QColor(0xff, 0xff, 0xff));
  const QRect r = QRect(50, 247, 100, 50);
  p.drawText(r, Qt::AlignCenter, net_type);

  // metrics
  drawMetric(p, temp_status.first, temp_status.second, 338);
  drawMetric(p, panda_status.first, panda_status.second, 496);
  drawMetric(p, connect_status.first, connect_status.second, 654);
}
