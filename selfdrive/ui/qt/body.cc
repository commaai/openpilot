#include "selfdrive/ui/qt/body.h"

#include <cmath>

#include <QPainter>

BodyWindow::BodyWindow(QWidget *parent) : QLabel(parent) {
  awake = new QMovie("../assets/body/awake.gif");
  awake->setCacheMode(QMovie::CacheAll);
  sleep = new QMovie("../assets/body/sleep.gif");
  sleep->setCacheMode(QMovie::CacheAll);

  QPalette p(Qt::black);
  setPalette(p);
  setAutoFillBackground(true);

  setAlignment(Qt::AlignCenter);

  setAttribute(Qt::WA_TransparentForMouseEvents, true);

  QObject::connect(uiState(), &UIState::uiUpdate, this, &BodyWindow::updateState);
}

void BodyWindow::paintEvent(QPaintEvent *event) {
  QLabel::paintEvent(event);

  QPainter p(this);
  p.setRenderHint(QPainter::Antialiasing);
  p.setPen(Qt::NoPen);

  // draw battery level
  const int offset = 90;
  const int radius = 60 / 2;
  const int levels = 5;
  const float interval = 1. / levels;
  for (int i = 0; i < levels; i++) {
    float level = 1.0 - (i+1)*interval;
    float perc = (fuel >= level) ? 1.0 : 0.35;

    p.setBrush(QColor(255, 255, 255, 255 * perc));
    QPoint pt(width() - (i*offset + offset / 2), offset / 2);
    p.drawEllipse(pt, radius, radius);
  }
}


void BodyWindow::updateState(const UIState &s) {
  if (!isVisible()) {
    return;
  }

  const SubMaster &sm = *(s.sm);
  auto cs = sm["carState"].getCarState();

  fuel = cs.getFuelGauge();

  // TODO: use carState.standstill when that's fixed
  const bool standstill = std::abs(cs.getVEgo()) < 0.01;
  QMovie *m = standstill ? sleep : awake;
  if (m != movie()) {
    setMovie(m);
    movie()->start();
  }

  update();
}
