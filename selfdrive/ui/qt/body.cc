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
  const SubMaster &sm = *(s.sm);

  // TODO: use standstill when that's fixed
  QMovie *m = sm["carState"].getCarState().getVEgoRaw() < 0.2 ? sleep : awake;
  if (m != movie()) {
    movie()->stop();
    setMovie(m);
    movie()->start();
  }
}
