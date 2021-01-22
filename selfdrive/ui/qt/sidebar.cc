#include "sidebar.hpp"
#include "widgets/toggle.hpp"
#include <QDebug>

StatusWidget::StatusWidget(QString text, QWidget *parent) : QFrame(parent)
{
  qDebug() << "StatusWidget ctor";
  
  QLabel* label = new QLabel(this);
  label->setText(text);
  label->setWordWrap(true);
  label->setStyleSheet(R"(font-size: 35px; font-weight: 500;background-color: blue;)");
  label->setAlignment(Qt::AlignVCenter | Qt::AlignHCenter);
  // label->setContentsMargins(50,50,50,50);
  // label->setMaximumSize(190,150-12-32);
  
  
  QHBoxLayout* sw_layout = new QHBoxLayout();
  sw_layout->setSpacing(0);
  // sw_layout->setContentsMargins(32,16,32,16);
  // sw_layout->setSpacing(20);

  // sw_layout->setMargin(32);
  // setFixedSize(300,150);
  // setMinimumSize(300,200);

  sw_layout->addWidget(label, 0, Qt::AlignTop | Qt::AlignLeft);

  setStyleSheet(R"( StatusWidget { background-color: red;})");

  setLayout(sw_layout);

}

void StatusWidget::paintEvent(QPaintEvent *e){
  //TODO: dynamically calculate strokes/borders
  // this->setFixedHeight(128); 
  // this->setFixedWidth(236);
  // this->setFixedHeight(200); 
  // this->setFixedWidth(300);
  QPainter p(this);
  p.setRenderHint(QPainter::Antialiasing, true);
  
  p.setPen(QPen(Qt::white, 3, Qt::SolidLine, Qt::FlatCap));
  p.setBrush(Qt::black);
  //origin at 1.5,1.5 because borders and qt issues with pixel perfect
  //less 6px for 2 * 3px borders
  // p.drawRoundedRect(QRectF(1.5+32, 1.5+16, 236-6, 128-6), 30, 30);
  // p.drawRoundedRect(QRectF(1.5, 1.5, 236, 128), 30, 30);
  p.drawRoundedRect(QRectF(1.5, 1.5, size().width()-3, size().height()-3), 30, 30);

  
  // draw indicator background
  p.setPen(Qt::NoPen);
  p.setBrush(Qt::white);
  p.setClipRect(0,0,25+6,size().height()-6,Qt::ClipOperation::ReplaceClip);
  // p.drawRoundedRect(QRectF(6+32, 6+16, size().width()-15, size().height()-15), 25, 25);
  p.drawRoundedRect(QRectF(6, 6, size().width()-12, size().height()-12), 25, 25);

  qDebug() << "sw size:"<< size();
}


Sidebar::Sidebar(QWidget *parent) : QFrame(parent){
  qDebug() << "Sidebar ctor";

  QVBoxLayout* main_layout = new QVBoxLayout();
  main_layout->setContentsMargins(32,16,32,16);
  main_layout->setSpacing(0);
  setFixedSize(300,1080);


  StatusWidget* sw1 = new StatusWidget("VEHICLE\nGOOD GPS", this);
  // sw1->indicator->setVisible(false);
  StatusWidget* sw2 = new StatusWidget("CONNECT\nOFFLINE", this);

  main_layout->addWidget(sw2, 0,Qt::AlignTop);
  main_layout->addWidget(sw1, 0,Qt::AlignTop);
  // QLabel* date = new QLabel(this);
  // date->setText("TESTTEST");
  
  // main_layout->addWidget(date, 1, Qt::AlignTop | Qt::AlignLeft);

  // setMaximumHeight(500);
  // setMaximumWidth(1000);
  // setFixedSize(400,400);
  
  // setStyleSheet(R"(background-color: red; border: 1px solid red;)");
  setStyleSheet(R"( Sidebar { background: qlineargradient(
                                    x1: 0, y1: 0, x2: 0, y2: 1,
                                    stop: 0 #242424,
                                    stop: 1.0 #030303);
                                    } )");
  setLayout(main_layout);
}