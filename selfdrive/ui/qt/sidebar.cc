#include "common/util.h"
#include "sidebar.h"
#include "qt_window.h"
#include "selfdrive/hardware/hw.h"

/*
void StatusWidget::paintEvent(QPaintEvent *e) {
  QPainter p(this);
  p.setRenderHint(QPainter::Antialiasing, true);
  p.setPen(QPen(QColor(0xb2b2b2), 3, Qt::SolidLine, Qt::FlatCap));
  // origin at 1.5,1.5 because qt issues with pixel perfect borders
  p.drawRoundedRect(QRectF(1.5, 1.5, size().width()-3, size().height()-3), 30, 30);

  p.setPen(Qt::NoPen);
  p.setBrush(color);
  p.setClipRect(0,0,25+6,size().height()-6,Qt::ClipOperation::ReplaceClip);
  p.drawRoundedRect(QRectF(6, 6, size().width()-12, size().height()-12), 25, 25);
}
*/

Sidebar::Sidebar(QWidget *parent) : QFrame(parent) {
  /*
  QVBoxLayout *layout = new QVBoxLayout();

  QPushButton *s_btn = new QPushButton;
  s_btn->setStyleSheet(R"(
    border-image: url(../assets/images/button_settings.png);
  )");
  s_btn->setFixedSize(200, 117);
  layout->addWidget(s_btn, 0, Qt::AlignHCenter);
  QObject::connect(s_btn, &QPushButton::pressed, this, &Sidebar::openSettings);

  s_btn->move(50, 35);
  */

  home_img.load("../assets/images/button_home.png");
  settings_img.load("../assets/images/button_settings.png");

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

  static std::map<cereal::DeviceState::NetworkType, const char *> network_type_map = {
      {cereal::DeviceState::NetworkType::NONE, "--"},
      {cereal::DeviceState::NetworkType::WIFI, "WiFi"},
      {cereal::DeviceState::NetworkType::CELL2_G, "2G"},
      {cereal::DeviceState::NetworkType::CELL3_G, "3G"},
      {cereal::DeviceState::NetworkType::CELL4_G, "4G"},
      {cereal::DeviceState::NetworkType::CELL5_G, "5G"}};
  const char *network_type = network_type_map[s.scene.deviceState.getNetworkType()];
  static std::map<cereal::DeviceState::NetworkStrength, int> network_strength_map = {
      {cereal::DeviceState::NetworkStrength::UNKNOWN, 1},
      {cereal::DeviceState::NetworkStrength::POOR, 2},
      {cereal::DeviceState::NetworkStrength::MODERATE, 3},
      {cereal::DeviceState::NetworkStrength::GOOD, 4},
      {cereal::DeviceState::NetworkStrength::GREAT, 5}};
  const int img_idx = s.scene.deviceState.getNetworkType() == cereal::DeviceState::NetworkType::NONE ? 0 : network_strength_map[s.scene.deviceState.getNetworkStrength()];
  signal->update(network_type, img_idx);

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
  p.setBrush(QBrush(QColor(0x39, 0x39, 0x39, 0xff)));

  // draw settings button
  p.setOpacity(0.65);
  p.drawImage(settings_btn.x(), settings_btn.y(), settings_img);

  // draw home button
  p.setOpacity(1.0);
  p.drawImage(60, 1080 - 180 - 40, home_img);

  //p.setBrush(QBrush(QColor(0x39, 0x39, 0x39, 0xff)));

  // network signal

  // network type

  // temperature


  // panda



  // connect


  // home button

}
