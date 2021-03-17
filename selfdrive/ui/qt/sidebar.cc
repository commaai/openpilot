#include "common/util.h"
#include "sidebar.hpp"
#include "ui.hpp"
#include "widgets/toggle.hpp"

//TODO: indicator widgets signals (text and color)

StatusWidget::StatusWidget(QString label, QString msg, QColor indicator, QWidget* parent) : QFrame(parent),
_severity(indicator)
{ l_label = new QLabel(this);
  l_label->setText(label);

  QVBoxLayout* sw_layout = new QVBoxLayout();
  sw_layout->setSpacing(0);

  if(msg.length() > 0){
    sw_layout->setContentsMargins(50,24,16,24); //50l, 16r, 24 vertical
    l_label->setStyleSheet(R"(font-size: 65px; font-weight: 600;)");
    l_label->setAlignment(Qt::AlignLeft | Qt::AlignHCenter);

		l_msg = new QLabel(this);
    l_msg->setText(msg);
    l_msg->setStyleSheet(R"(font-size: 30px; font-weight: 400;)");
    l_msg->setAlignment(Qt::AlignLeft | Qt::AlignHCenter);

    sw_layout->addWidget(l_label, 0, Qt::AlignLeft);
    sw_layout->addWidget(l_msg, 0, Qt::AlignLeft);
  } else {
    sw_layout->setContentsMargins(40,24,16,24); //40l, 16r, 24 vertical

    l_label->setStyleSheet(R"(font-size: 30px; font-weight: 600;)");
    l_label->setAlignment(Qt::AlignVCenter | Qt::AlignHCenter);

    sw_layout->addWidget(l_label, 0, Qt::AlignCenter);
  }

  setMinimumHeight(124);
  setMaximumSize(300,200);
  setStyleSheet(R"( StatusWidget { background-color: transparent;})");
  setLayout(sw_layout);
}

void StatusWidget::paintEvent(QPaintEvent *e){
  QPainter p(this);
  p.setRenderHint(QPainter::Antialiasing, true);
  p.setPen(QPen(QColor(0xb2b2b2), 3, Qt::SolidLine, Qt::FlatCap));
  //origin at 1.5,1.5 because qt issues with pixel perfect borders
  p.drawRoundedRect(QRectF(1.5, 1.5, size().width()-3, size().height()-3), 30, 30);

  p.setPen(Qt::NoPen);
  p.setBrush(_severity);
  p.setClipRect(0,0,25+6,size().height()-6,Qt::ClipOperation::ReplaceClip);
  p.drawRoundedRect(QRectF(6, 6, size().width()-12, size().height()-12), 25, 25);
}

void StatusWidget::update(QString label, QString msg, QColor color){
	l_label->setText(label);
	if(msg.length() > 0)
		l_msg->setText(msg);
	_severity = color;
	return;
}

SignalWidget::SignalWidget(QString text, int strength, QWidget* parent) : QFrame(parent),
_strength(strength)
{

  label = new QLabel(text, this);
  label->setStyleSheet(R"(font-size: 35px; font-weight: 200;)");
  label->setAlignment(Qt::AlignVCenter);
  label->setFixedSize(100, 50);
  QVBoxLayout* layout = new QVBoxLayout();
  layout->setSpacing(0);
  layout->insertSpacing(0,35);
  layout->setContentsMargins(50,0,50,0);
  layout->addWidget(label, 0, Qt::AlignVCenter | Qt::AlignLeft);
  setMinimumHeight(120);
  setMaximumSize(176,80);
  setLayout(layout);
}

void SignalWidget::paintEvent(QPaintEvent *e){
  QPainter p(this);
  p.setRenderHint(QPainter::Antialiasing, true);
  p.setPen(Qt::NoPen);
  p.setBrush(Qt::white);
  for (int i = 0; i < 5 ; i++) //draw empty dots
  {
    if(i == _strength)
    {
      p.setPen(Qt::NoPen);
      p.setBrush(Qt::darkGray);
    }
    p.drawEllipse(QRectF(_dotspace * i, _top, _dia, _dia));
  }
}

void SignalWidget::update(QString text, int strength){
  if(text.length() > 0)
    label->setText(text);
  _strength = strength;
}

Sidebar::Sidebar(QWidget* parent) : QFrame(parent){
  QVBoxLayout* sb_layout = new QVBoxLayout();
  sb_layout->setContentsMargins(24,16,24,16); //24 sides, 16 vertical
  sb_layout->setSpacing(16);
  setFixedSize(300,1080);

  QImage image = QImageReader("../assets/images/button_home.png").read();
  QLabel *comma = new QLabel();
  comma->setPixmap(QPixmap::fromImage(image));
  comma->setAlignment(Qt::AlignCenter);
  comma->setFixedSize(200,200);

  // TODO: set opacity to 0.65
  QPushButton* s_btn = new QPushButton(this);

  //  border-image: url(../assets/images/button_settings.png);
  s_btn->setStyleSheet(R"(
    background-color: rgba(0, 0, 0, 0);
    background-image: url(../assets/images/button_settings.png);
    background-repeat: no-repeat;
    background-position: center;
  )");
  s_btn->setFixedSize(252, 150);


  signal = new SignalWidget("4G",2,this);
  temp = new StatusWidget("39C", "TEMP", QColor(201, 34, 49), this);
  vehicle = new StatusWidget("VEHICLE\nGOOD GPS", "", QColor(255, 255, 255), this);
  connect = new StatusWidget("CONNECT\nOFFLINE", "",  QColor(218, 202, 37), this);


  sb_layout->addWidget(s_btn, 0, Qt::AlignTop);
  sb_layout->addWidget(signal, 0, Qt::AlignTop | Qt::AlignHCenter);
  sb_layout->addWidget(temp, 0, Qt::AlignTop);
  sb_layout->addWidget(vehicle, 0, Qt::AlignTop);
  sb_layout->addWidget(connect, 0, Qt::AlignTop);
  sb_layout->addStretch(1);
  sb_layout->addWidget(comma, 0, Qt::AlignHCenter | Qt::AlignVCenter);
  setStyleSheet(R"( Sidebar { background-color: #393939 ;})");
  setLayout(sb_layout);
}

void Sidebar::update(UIState *s){
  static std::map<NetStatus, std::pair<const char *, int>> connectivity_map = {
    {NET_ERROR, {"CONNECT\nERROR", 2}},
    {NET_CONNECTED, {"CONNECT\nONLINE", 0}},
    {NET_DISCONNECTED, {"CONNECT\nOFFLINE", 1}},
  };
  auto net_params = connectivity_map[s->scene.athenaStatus];
	connect->update(net_params.first, "", QColor(218, 202, 37));

	static std::map<cereal::DeviceState::ThermalStatus, const int> temp_severity_map = {
				{cereal::DeviceState::ThermalStatus::GREEN, 0},
				{cereal::DeviceState::ThermalStatus::YELLOW, 1},
				{cereal::DeviceState::ThermalStatus::RED, 2},
				{cereal::DeviceState::ThermalStatus::DANGER, 3}};
	std::string temp_val = std::to_string((int)s->scene.deviceState.getAmbientTempC()) + "°C";
	temp->update(QString(temp_val.c_str()), "", QColor(201, 34, 49));

	static std::map<cereal::DeviceState::NetworkType, const char *> network_type_map = {
      {cereal::DeviceState::NetworkType::NONE, "--"},
      {cereal::DeviceState::NetworkType::WIFI, "WiFi"},
      {cereal::DeviceState::NetworkType::CELL2_G, "2G"},
      {cereal::DeviceState::NetworkType::CELL3_G, "3G"},
      {cereal::DeviceState::NetworkType::CELL4_G, "4G"},
      {cereal::DeviceState::NetworkType::CELL5_G, "5G"}};
  const char *network_type = network_type_map[s->scene.deviceState.getNetworkType()];
  static std::map<cereal::DeviceState::NetworkStrength, int> network_strength_map = {
      {cereal::DeviceState::NetworkStrength::UNKNOWN, 1},
      {cereal::DeviceState::NetworkStrength::POOR, 2},
      {cereal::DeviceState::NetworkStrength::MODERATE, 3},
      {cereal::DeviceState::NetworkStrength::GOOD, 4},
      {cereal::DeviceState::NetworkStrength::GREAT, 5}};
  const int img_idx = s->scene.deviceState.getNetworkType() == cereal::DeviceState::NetworkType::NONE ? 0 : network_strength_map[s->scene.deviceState.getNetworkStrength()];
  signal->update(network_type, img_idx);

  int panda_severity = 0;
  std::string panda_message = "VEHICLE\nONLINE";
  if (s->scene.pandaType == cereal::PandaState::PandaType::UNKNOWN) {
    panda_severity = 2;
    panda_message = "NO\nPANDA";
  } else if (s->scene.started) {
    panda_severity = s->scene.gpsOK ? 0 : 1;
    panda_message = util::string_format("SAT CNT\n%d", s->scene.satelliteCount);
  }

  vehicle->update(panda_message.c_str(), "", QColor(255, 255, 255));
}
