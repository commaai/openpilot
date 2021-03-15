#include "sidebar.hpp"
#include "widgets/toggle.hpp"

//TODO: settings btn signals
//TODO: indicator widgets signals (text and color)

SettingsBtn::SettingsBtn(QWidget* parent) : QAbstractButton(parent){
  setMinimumHeight(160);
  setMaximumSize(300,200);
  image = QImageReader("../assets/images/button_settings.png").read().scaledToWidth(200,Qt::SmoothTransformation);
}

void SettingsBtn::paintEvent(QPaintEvent *e){
  QPainter p(this);
  p.setOpacity(0.65);
  p.drawImage((size().width()/2)-(image.width()/2), (size().height()/2)-(image.height()/2),
              image, 0 , 0 , 0 , 0, Qt::AutoColor);
}


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


SignalWidget::SignalWidget(QString text, int strength, QWidget* parent) : QFrame(parent),
_strength(strength)
{

  QLabel* label = new QLabel(text, this);
  label->setStyleSheet(R"(font-size: 35px; font-weight: 200;)");
  label->setAlignment(Qt::AlignVCenter);
  label->setFixedSize(100, 50);
  QVBoxLayout* layout = new QVBoxLayout();
  layout->setSpacing(0);
  layout->insertSpacing(0,35);
  layout->setContentsMargins(50,0,50,0);
  layout->addWidget(label, 0, Qt::AlignVCenter | Qt::AlignLeft);
  setMinimumHeight(120);
  //setMaximumSize(300,150);
  setMaximumSize(176,80);
  setLayout(layout);
}

void SignalWidget::paintEvent(QPaintEvent *e){
  QPainter p(this);
  p.setRenderHint(QPainter::Antialiasing, true);
  p.setPen(Qt::NoPen);
  p.setBrush(Qt::darkGray);
  for (int i = _strength; i < 5 ; i++) //draw empty dots
  {
    p.drawEllipse(QRectF(_dotspace * i, _top, _dia, _dia));
  }
  p.setPen(Qt::NoPen);
  p.setBrush(Qt::white);
  for (int i = 0; i < _strength; i++) //draw filled dots
  {
    p.drawEllipse(QRectF(_dotspace * i, _top, _dia, _dia));
  }
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

  SettingsBtn* s_btn = new SettingsBtn(this);
  SignalWidget* signal = new SignalWidget("4G",2,this);
  //TODO: better temp widget layouting/font
  StatusWidget* temp = new StatusWidget("39C", "TEMP", QT_COLOR_RED, this);
  StatusWidget* vehicle = new StatusWidget("VEHICLE\nGOOD GPS", "", QT_COLOR_WHITE, this);
  StatusWidget* connect = new StatusWidget("CONNECT\nOFFLINE", "",  QT_COLOR_YELLOW, this);


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
