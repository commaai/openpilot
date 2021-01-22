#include "sidebar.hpp"
#include "widgets/toggle.hpp"
#include <QDebug>

StatusWidget::StatusWidget(QString text, QWidget *parent) : QFrame(parent)
{
  qDebug() << "StatusWidget ctor";
  
  QLabel* label = new QLabel(this);
  label->setText(text);
  label->setStyleSheet(R"(font-size: 35px; font-weight: 600;)");
  label->setAlignment(Qt::AlignVCenter | Qt::AlignHCenter);  

  QHBoxLayout* sw_layout = new QHBoxLayout();
  sw_layout->setSpacing(0);
  sw_layout->setContentsMargins(40,16,16,16);
  sw_layout->addWidget(label, 0, Qt::AlignCenter);

  setMinimumHeight(120);
  setMaximumSize(300,200);
  setStyleSheet(R"( StatusWidget { background-color: transparent;})");
  setLayout(sw_layout);


  qDebug() << "sw size ctor: "<< size();  

}

void StatusWidget::paintEvent(QPaintEvent *e){
  QPainter p(this);
  p.setRenderHint(QPainter::Antialiasing, true);
  
  p.setPen(QPen(QColor(0xb2b2b2), 3, Qt::SolidLine, Qt::FlatCap));
  p.setBrush(Qt::black);
  //origin at 1.5,1.5 because borders and qt issues with pixel perfect
  //less 6px for 2 * 3px borders
  p.drawRoundedRect(QRectF(1.5, 1.5, size().width()-3, size().height()-3), 30, 30);


  QLinearGradient gradient = QLinearGradient(0,0,0,100);
  gradient.setColorAt(0.0, QColor(0xcfcfcf));
  gradient.setColorAt(1.0, QColor(0xb2b2b2));
  QBrush brush(gradient);
  
  // draw indicator background
  p.setPen(Qt::NoPen);
  p.setBrush(brush);
  p.setClipRect(0,0,25+6,size().height()-6,Qt::ClipOperation::ReplaceClip);
  p.drawRoundedRect(QRectF(6, 6, size().width()-12, size().height()-12), 25, 25);

  qDebug() << "sw size:"<< size();
}


Sidebar::Sidebar(QWidget *parent) : QFrame(parent){
  qDebug() << "Sidebar ctor";

  QVBoxLayout* main_layout = new QVBoxLayout();
  main_layout->setContentsMargins(24,16,24,16);
  main_layout->setSpacing(0);
  setFixedSize(300,1080);

  StatusWidget* sw1 = new StatusWidget("VEHICLE\nGOOD GPS", this);
  // sw1->indicator->setVisible(false);
  StatusWidget* sw2 = new StatusWidget("CONNECT\nOFFLINE", this);

  main_layout->addWidget(sw2, 0,Qt::AlignTop);
  main_layout->addWidget(sw1, 0,Qt::AlignTop);
  setStyleSheet(R"( Sidebar { background: qlineargradient(
                                    x1: 0, y1: 0, x2: 0, y2: 1,
                                    stop: 0 #242424,
                                    stop: 1.0 #030303);
                                    } )");
  setLayout(main_layout);
}