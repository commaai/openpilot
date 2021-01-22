#include "sidebar.hpp"
#include "widgets/toggle.hpp"
#include <QDebug>

//TODO: custom qlabel text size

SettingsBtn::SettingsBtn(QWidget* parent) : QAbstractButton(parent){
  setMinimumHeight(160);
  setMaximumSize(300,200);
  image = QImageReader("../assets/images/button_settings.png").read().scaledToWidth(110,Qt::SmoothTransformation);
}

void SettingsBtn::paintEvent(QPaintEvent *e){
  QPainter p(this);
  //New settings btn stuff
  p.setRenderHint(QPainter::Antialiasing, true);
  p.setPen(QPen(QColor(0xb2b2b2), 3, Qt::SolidLine, Qt::FlatCap));
  p.setBrush(Qt::black);
  // origin at 1.5,1.5 because qt issues with pixel perfect borders
  p.drawRoundedRect(QRectF(1.5, 1.5, size().width()-3, size().height()-3), 30, 30);
  p.drawImage((size().width()/2)-(image.width()/2), (size().height()/2)-(image.height()/2), image,0,0,0,0, Qt::AutoColor);
}

StatusWidget::StatusWidget(QString text, QColor indicator, QWidget* parent) : QFrame(parent),
ind_color(indicator)
{  
  QLabel* label = new QLabel(this);
  label->setText(text);
  label->setStyleSheet(R"(font-size: 35px; font-weight: 600;)");
  label->setAlignment(Qt::AlignVCenter | Qt::AlignHCenter);  

  QHBoxLayout* sw_layout = new QHBoxLayout();
  sw_layout->setSpacing(0);
  sw_layout->setContentsMargins(40,24,16,24); //40l, 16r, 24 vertical
  sw_layout->addWidget(label, 0, Qt::AlignCenter);

  setMinimumHeight(120);
  setMaximumSize(300,200);
  setStyleSheet(R"( StatusWidget { background-color: transparent;})");
  setLayout(sw_layout);
}

void StatusWidget::paintEvent(QPaintEvent *e){
  QPainter p(this);
  p.setRenderHint(QPainter::Antialiasing, true);  
  p.setPen(QPen(QColor(0xb2b2b2), 3, Qt::SolidLine, Qt::FlatCap));
  p.setBrush(Qt::black);
  //origin at 1.5,1.5 because qt issues with pixel perfect borders
  p.drawRoundedRect(QRectF(1.5, 1.5, size().width()-3, size().height()-3), 30, 30);

  // TODO: reimplement gradient indicators
  // TODO: postponed: indicator highlight and blinking?
  // draw indicator background
  // QLinearGradient gradient = QLinearGradient(0,0,0,100);
  // gradient.setColorAt(0.0, QColor(0xcfcfcf));
  // gradient.setColorAt(1.0, QColor(0xb2b2b2));
  // QBrush brush(gradient);
  
  p.setPen(Qt::NoPen);
  p.setBrush(ind_color);
  p.setClipRect(0,0,25+6,size().height()-6,Qt::ClipOperation::ReplaceClip);
  p.drawRoundedRect(QRectF(6, 6, size().width()-12, size().height()-12), 25, 25);
}


SignalWidget::SignalWidget(QString text, int strength, QWidget* parent) : QFrame(parent),
_strength(strength)
{
  setFixedSize(300,80);
  QLabel* label = new QLabel(this);
  label->setText(text);
  label->setStyleSheet(R"(font-size: 30px; font-weight: 600;)");
  label->setAlignment(Qt::AlignVCenter);
  QHBoxLayout* layout = new QHBoxLayout();
  layout->setSpacing(0);
  layout->setContentsMargins(24,16,24,16);
  layout->addWidget(label, 0, Qt::AlignLeft);
  setLayout(layout);
}

void SignalWidget::paintEvent(QPaintEvent *e){  
  QPainter p(this);
  p.setRenderHint(QPainter::Antialiasing, true);
  p.setPen(QPen(QColor(0xb2b2b2), 3, Qt::SolidLine, Qt::FlatCap));
  for (int i = _strength; i < 5 ; i++) //draw empty dots
  {
    p.drawEllipse(QRectF(1.5 + _firstdot + _dotspace * i, 1.5 + _top, _dia - 3, _dia - 3));
  }
  p.setPen(Qt::NoPen);
  p.setBrush(Qt::white);
  for (int i = 0; i < _strength; i++) //draw filled dots
  {
    p.drawEllipse(QRectF(_firstdot + _dotspace * i, _top, _dia, _dia));
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
  StatusWidget* temp = new StatusWidget("39C\nTEMP", Qt::white, this); //test white
  StatusWidget* vehicle = new StatusWidget("VEHICLE\nGOOD GPS", QColor(0x33ab4c), this); //test green
  StatusWidget* connect = new StatusWidget("CONNECT\nOFFLINE",  QColor(234,192,11), this); //test yellow
  

  sb_layout->addWidget(signal, 0, Qt::AlignTop);
  sb_layout->addWidget(s_btn, 0, Qt::AlignTop);
  sb_layout->addWidget(temp, 0, Qt::AlignTop);
  sb_layout->addWidget(vehicle, 0, Qt::AlignTop);
  sb_layout->addWidget(connect, 0, Qt::AlignTop);
  sb_layout->addStretch(1);
  sb_layout->addWidget(comma, 0, Qt::AlignHCenter | Qt::AlignVCenter);
  setStyleSheet(R"( Sidebar { background: qlineargradient(
                                    x1: 0, y1: 0, x2: 0, y2: 1,
                                    stop: 0 #242424,
                                    stop: 1.0 #030303);
                                    } )");
  setLayout(sb_layout);
}
