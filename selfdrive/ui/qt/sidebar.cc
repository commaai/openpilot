#include "common/util.h"
#include "sidebar.h"
#include "qt_window.h"

StatusWidget::StatusWidget(QString label, QString msg, QColor c, QWidget* parent) : QFrame(parent) {
  layout.setSpacing(0);

  if(msg.length() > 0){
    layout.setContentsMargins(50, 24, 16, 24);
    status.setAlignment(Qt::AlignLeft | Qt::AlignHCenter);
    status.setStyleSheet(R"(font-size: 65px; font-weight: 500;)");

    substatus.setAlignment(Qt::AlignLeft | Qt::AlignHCenter);
    substatus.setStyleSheet(R"(font-size: 30px; font-weight: 400;)");

    layout.addWidget(&status, 0, Qt::AlignLeft);
    layout.addWidget(&substatus, 0, Qt::AlignLeft);
  } else {
    layout.setContentsMargins(40, 24, 16, 24);

    status.setAlignment(Qt::AlignCenter);
    status.setStyleSheet(R"(font-size: 38px; font-weight: 500;)");
    layout.addWidget(&status, 0, Qt::AlignCenter);
  }

  update(label, msg, c);

  setMinimumHeight(124);
  setStyleSheet("background-color: transparent;");
  setLayout(&layout);
}

void StatusWidget::paintEvent(QPaintEvent *e){
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

void StatusWidget::update(QString label, QString msg, QColor c) {
  status.setText(label);
  substatus.setText(msg);

  if (color != c) {
    color = c;
    repaint();
  }
  return;
}

SignalWidget::SignalWidget(QString text, int strength, QWidget* parent) : QFrame(parent), _strength(strength) {
  layout.setMargin(0);
  layout.setSpacing(0);
  layout.insertSpacing(0, 45);

  label.setText(text);
  layout.addWidget(&label, 0, Qt::AlignLeft);
  label.setStyleSheet(R"(font-size: 35px; font-weight: 400;)");

  setFixedWidth(177);
  setLayout(&layout);
}

void SignalWidget::paintEvent(QPaintEvent *e){
  QPainter p(this);
  p.setRenderHint(QPainter::Antialiasing, true);
  p.setPen(Qt::NoPen);
  p.setBrush(Qt::white);
  for (int i = 0; i < 5 ; i++){
    if(i == _strength){
      p.setPen(Qt::NoPen);
      p.setBrush(Qt::darkGray);
    }
    p.drawEllipse(QRectF(_dotspace * i, _top, _dia, _dia));
  }
}

void SignalWidget::update(QString text, int strength){
  label.setText(text);
  _strength = strength;
}

Sidebar::Sidebar(QWidget* parent) : QFrame(parent) {
  QVBoxLayout* layout = new QVBoxLayout();
  layout->setContentsMargins(25, 50, 25, 50);
  layout->setSpacing(35);
  setFixedSize(300, vwp_h);

  QPushButton *s_btn = new QPushButton;
  s_btn->setStyleSheet(R"(
    border-image: url(../assets/images/button_settings.png);
  )");
  s_btn->setFixedSize(200, 117);
  layout->addWidget(s_btn, 0, Qt::AlignHCenter);
  QObject::connect(s_btn, &QPushButton::pressed, this, &Sidebar::openSettings);

  signal = new SignalWidget("--", 0, this);
  layout->addWidget(signal, 0, Qt::AlignTop | Qt::AlignHCenter);

  temp = new StatusWidget("0°C", "TEMP", QColor(255, 255, 255), this);
  layout->addWidget(temp, 0, Qt::AlignTop);

  panda = new StatusWidget("NO\nPANDA", "", QColor(201, 34, 49), this);
  layout->addWidget(panda, 0, Qt::AlignTop);

  connect = new StatusWidget("CONNECT\nOFFLINE", "",  QColor(218, 202, 37), this);
  layout->addWidget(connect, 0, Qt::AlignTop);

  QImage image = QImageReader("../assets/images/button_home.png").read();
  QLabel *comma = new QLabel(this);
  comma->setPixmap(QPixmap::fromImage(image));
  comma->setAlignment(Qt::AlignCenter);
  layout->addWidget(comma, 1, Qt::AlignHCenter | Qt::AlignVCenter);

  layout->addStretch(1);

  setStyleSheet(R"(
    Sidebar {
      background-color: #393939;
    }
    * {
      color: white;
    }
  )");
  setLayout(layout);
}

void Sidebar::update(const UIState &s){
  static std::map<NetStatus, std::pair<QString, QColor>> connectivity_map = {
    {NET_ERROR, {"CONNECT\nERROR", COLOR_DANGER}},
    {NET_CONNECTED, {"CONNECT\nONLINE", COLOR_GOOD}},
    {NET_DISCONNECTED, {"CONNECT\nOFFLINE", COLOR_WARNING}},
  };
  auto net_params = connectivity_map[s.scene.athenaStatus];
  connect->update(net_params.first, "", net_params.second);

  static std::map<cereal::DeviceState::ThermalStatus, QColor> temp_severity_map = {
        {cereal::DeviceState::ThermalStatus::GREEN, COLOR_GOOD},
        {cereal::DeviceState::ThermalStatus::YELLOW, COLOR_WARNING},
        {cereal::DeviceState::ThermalStatus::RED, COLOR_DANGER},
        {cereal::DeviceState::ThermalStatus::DANGER, COLOR_DANGER}};
  QString temp_val = QString("%1 °C").arg((int)s.scene.deviceState.getAmbientTempC());
  temp->update(temp_val, "TEMP", temp_severity_map[s.scene.deviceState.getThermalStatus()]);

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
#ifdef QCOM2
  else if (s.scene.started) {
    panda_color = s.scene.gpsOK ? COLOR_GOOD : COLOR_WARNING;
    panda_message = QString("SAT CNT\n%1").arg(s.scene.satelliteCount);
  }
#endif
  panda->update(panda_message, "", panda_color);
}
