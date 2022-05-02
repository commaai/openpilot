#include <cmath>

#include <QGridLayout>
#include <QPainter>

#include "selfdrive/ui/qt/body.h"


FaceDotMatrix::FaceDotMatrix(QWidget* parent) : QWidget(parent) {
  QPalette pal = QPalette();
  pal.setColor(QPalette::Window, Qt::black);
  setAutoFillBackground(true); 
  setPalette(pal);
  setAttribute(Qt::WA_TransparentForMouseEvents, true);
}

void FaceDotMatrix::paintEvent(QPaintEvent *e) {
  QPainter painter(this);
  QPen linepen(Qt::white);
  linepen.setCapStyle(Qt::RoundCap);
  linepen.setWidth(30);
  painter.setRenderHint(QPainter::Antialiasing,true);
  painter.setPen(linepen);
  painter.drawPoint(width()/2, height()/2);
}

Eye::Eye(QWidget* parent) : FaceDotMatrix(parent) {
}

void Eye::paintEvent(QPaintEvent *e) {
  QPainter painter(this);
  QPen linepen(Qt::white);
  linepen.setCapStyle(Qt::RoundCap);
  linepen.setWidth(56);
  painter.setRenderHint(QPainter::Antialiasing,true);
  painter.setPen(linepen);
  painter.drawPoint(width()/2+100, height()/2+100);
}

Mouth::Mouth(QWidget* parent) : FaceDotMatrix(parent) {
}

void Mouth::paintEvent(QPaintEvent *e) {
  printf("%d %d\n", height(), width());
  QPainter painter(this);
  QPen linepen(Qt::black);
  linepen.setCapStyle(Qt::RoundCap);
  linepen.setWidth(30);
  painter.setRenderHint(QPainter::Antialiasing,true);
  painter.setPen(linepen);
  painter.drawPoint(0,0);
}


BodyWindow::BodyWindow(QWidget *parent) : QWidget(parent) {
  layout = new QGridLayout(this);

  QPalette pal = QPalette();
  pal.setColor(QPalette::Window, Qt::black);
  setAutoFillBackground(true); 
  setPalette(pal);

  setAttribute(Qt::WA_TransparentForMouseEvents, true);

  leftEye = new Eye(this);
  rightEye = new Eye(this);
  mouth = new Mouth(this);

  layout->addItem(new QSpacerItem(1,1,QSizePolicy::Expanding,QSizePolicy::Expanding), 0, 0);
  layout->addItem(new QSpacerItem(1,1,QSizePolicy::Expanding,QSizePolicy::Expanding), 0, 1);
  layout->addItem(new QSpacerItem(1,1,QSizePolicy::Expanding,QSizePolicy::Expanding), 0, 2);
  layout->addItem(new QSpacerItem(1,1,QSizePolicy::Expanding,QSizePolicy::Expanding), 0, 3);
  layout->addItem(new QSpacerItem(1,1,QSizePolicy::Expanding,QSizePolicy::Expanding), 0, 4);
  
  layout->addItem(new QSpacerItem(1,1,QSizePolicy::Expanding,QSizePolicy::Expanding), 1, 0);
  layout->addWidget(leftEye,1,1);
  layout->addItem(new QSpacerItem(1,1,QSizePolicy::Expanding,QSizePolicy::Expanding), 1, 2);
  layout->addWidget(rightEye,1,3);
  layout->addItem(new QSpacerItem(1,1,QSizePolicy::Expanding,QSizePolicy::Expanding), 1, 4);

  layout->addItem(new QSpacerItem(1,1,QSizePolicy::Expanding,QSizePolicy::Expanding), 2, 0);
  layout->addItem(new QSpacerItem(1,1,QSizePolicy::Expanding,QSizePolicy::Expanding), 2, 1);
  layout->addWidget(mouth,2,2);
  layout->addItem(new QSpacerItem(1,1,QSizePolicy::Expanding,QSizePolicy::Expanding), 2, 3);
  layout->addItem(new QSpacerItem(1,1,QSizePolicy::Expanding,QSizePolicy::Expanding), 2, 4);
  
  

  QObject::connect(uiState(), &UIState::uiUpdate, this, &BodyWindow::updateState);
}

void BodyWindow::updateState(const UIState &s) {
  if (!isVisible()) {
    return;
  }

  // const SubMaster &sm = *(s.sm);

  // TODO: use carState.standstill when that's fixed
  //const bool standstill = std::abs(sm["carState"].getCarState().getVEgo()) < 0.01;
}
