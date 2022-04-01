#include "selfdrive/ui/qt/body.h"

#include <QDebug>

BodyWindow::BodyWindow(QWidget *parent) : QLabel(parent) {
  awake = new QMovie("../assets/body/awake.gif");
  awake->setCacheMode(QMovie::CacheAll);
  sleep = new QMovie("../assets/body/sleep.gif");
  sleep->setCacheMode(QMovie::CacheAll);

  setMovie(sleep);
  movie()->start();

  setScaledContents(true);

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
    movie()->stop();
    setMovie(m);
    movie()->start();
  }
}
