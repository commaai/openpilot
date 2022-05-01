#include <cmath>

#include <QGridLayout>
#include <QPainter>

#include "selfdrive/ui/qt/body.h"

BodyWindow::BodyWindow(QWidget *parent) : QWidget(parent) {
  layout = new QGridLayout(this);
  layout->setMargin(0);
  layout->setSpacing(0);

  //setAttribute(Qt::WA_TransparentForMouseEvents, true);

  battery = new QLabel();
  battery->setStyleSheet("QLabel { font-size: 200px; }");
  battery->setAlignment(Qt::AlignTop | Qt::AlignRight);

  layout->addWidget(battery,0,2);
  //layout->addWidget(rightEye,0,2);
  //layout->addWidget(leftEye,0,2);
  //layout->addWidget(mouth,0,2);

  QObject::connect(uiState(), &UIState::uiUpdate, this, &BodyWindow::updateState);
}

void BodyWindow::paintEvent(QPaintEvent *e) {
  QPainter painter(this);
  QPen linepen(Qt::red);
  linepen.setCapStyle(Qt::RoundCap);
  linepen.setWidth(30);
  painter.setRenderHint(QPainter::Antialiasing,true);
  painter.setPen(linepen);
  painter.drawPoint(200,200);
}

void BodyWindow::updateState(const UIState &s) {
  if (!isVisible()) {
    return;
  }

  const SubMaster &sm = *(s.sm);

  // TODO: use carState.standstill when that's fixed
  //const bool standstill = std::abs(sm["carState"].getCarState().getVEgo()) < 0.01;

  battery->setText(generateBatteryText(sm["carState"].getCarState().getFuelGauge()));
}

QString BodyWindow::generateBatteryText(float fuelGauge) {
  int batteryOkDots = round(5*fuelGauge);
  int batteryDotsRemain = 5 - batteryOkDots;
  QString batteryDotsText = "<font color=\"white\">";
  for (int i=0; i<batteryOkDots; i++)
    batteryDotsText+="·";
  batteryDotsText+="</font><font color=\"grey\">";
  for (int i=0; i<batteryDotsRemain; i++)
    batteryDotsText+="·";
  batteryDotsText+="</font>";
  return batteryDotsText;
}
