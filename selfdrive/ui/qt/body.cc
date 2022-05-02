#include <cmath>

#include <QGridLayout>
#include <QPainter>

#include "selfdrive/ui/qt/body.h"


FaceDotMatrix::FaceDotMatrix(QWidget* parent) : QWidget(parent) {
  QPalette pal = QPalette();
  pal.setColor(QPalette::Window, Qt::yellow);
  setAutoFillBackground(true); 
  setPalette(pal);
  setAttribute(Qt::WA_TransparentForMouseEvents, true);
}

void FaceDotMatrix::paintEvent(QPaintEvent *e) {}

void FaceDotMatrix::paintMatrix(int a[4][4]) {
  QPainter painter(this);
  QPen linepen(Qt::white);
  linepen.setCapStyle(Qt::RoundCap);
  linepen.setWidth(dotSize);
  painter.setRenderHint(QPainter::Antialiasing,true);
  painter.setPen(linepen);

  int m = (dotSize+dotMargin*2);
  int xOffset = (width()/2) - 2*m;
  int yOffset = (height()/2) - 2*m;

  for (int i = 0; i<4; i++) {
    painter.drawPoint(xOffset + i*m + m/2, yOffset + i*m + m/2);
  }

  
}

Eye::Eye(QWidget* parent) : FaceDotMatrix(parent) {
  dotSize = 64;
  dotMargin = 10;
}

void Eye::paintEvent(QPaintEvent *e) {
  int a[4][4] = {
    0,1,1,0,
    1,1,1,1,
    1,1,1,1,
    0,1,1,0
  };
  paintMatrix(a);
}

void Eye::updateState(const UIState &s) {
  update();
}


Mouth::Mouth(QWidget* parent) : FaceDotMatrix(parent) {
}

void Mouth::paintEvent(QPaintEvent *e) {
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
  QObject::connect(uiState(), &UIState::uiUpdate, leftEye, &Eye::updateState);
  QObject::connect(uiState(), &UIState::uiUpdate, rightEye, &Eye::updateState);
}

void BodyWindow::updateState(const UIState &s) {
  if (!isVisible()) {
    return;
  }

  // const SubMaster &sm = *(s.sm);

  // TODO: use carState.standstill when that's fixed
  //const bool standstill = std::abs(sm["carState"].getCarState().getVEgo()) < 0.01;
}
