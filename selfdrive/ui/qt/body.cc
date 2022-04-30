#include <cmath>

#include <QVBoxLayout>

#include "selfdrive/ui/qt/body.h"

BodyWindow::BodyWindow(QWidget *parent) : QWidget(parent) {
  layout = new QVBoxLayout(this);
  layout->setMargin(0);
  layout->setSpacing(0);

  face = new QLabel();

  awake = new QMovie("../assets/body/awake.gif");
  awake->setCacheMode(QMovie::CacheAll);
  sleep = new QMovie("../assets/body/sleep.gif");
  sleep->setCacheMode(QMovie::CacheAll);

  face->setAlignment(Qt::AlignCenter);
  face->setAttribute(Qt::WA_TransparentForMouseEvents, true);

  battery = new QLabel();
  battery->setStyleSheet("QLabel { font-size: 54px; }");
  battery->setAlignment(Qt::AlignBottom | Qt::AlignRight);
  battery->adjustSize();

  QPalette p(Qt::black);
  face->setPalette(p);
  face->setAutoFillBackground(true);

  layout->addWidget(face);
  layout->addWidget(battery);

  QObject::connect(uiState(), &UIState::uiUpdate, this, &BodyWindow::updateState);
}

void BodyWindow::updateState(const UIState &s) {
  if (!isVisible()) {
    return;
  }

  const SubMaster &sm = *(s.sm);

  // TODO: use carState.standstill when that's fixed
  const bool standstill = std::abs(sm["carState"].getCarState().getVEgo()) < 0.01;

  battery->setText("<font color=\"red\">🔋</font> <font color=\"white\">"+QString::number(sm["carState"].getCarState().getBatteryPercent())+"%</font>");

  QMovie *m = standstill ? sleep : awake;
  if (m != face->movie()) {
    face->setMovie(m);
    face->movie()->start();
  }
}
