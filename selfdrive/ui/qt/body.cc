#include "selfdrive/ui/qt/body.h"

#include <cmath>

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

void BodyWindow::updateState(const UIState &s) {
  if (!isVisible()) {
    return;
  }

  const SubMaster &sm = *(s.sm);

  // TODO: use carState.standstill when that's fixed
  const bool standstill = std::abs(sm["carState"].getCarState().getVEgo()) < 0.01;
  QMovie *m = standstill ? sleep : awake;
  if (m != movie()) {
    setMovie(m);
    movie()->start();
  }
}
