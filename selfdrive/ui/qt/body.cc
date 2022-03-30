#include "selfdrive/ui/qt/body.h"

#include <QLabel>
#include <QPainter>

BodyWindow::BodyWindow(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout  = new QVBoxLayout(this);

  // setup layouts + widgets here
  QLabel *l = new QLabel("body ui");
  l->setStyleSheet("color: white; font-size: 30px;");
  main_layout->addWidget(l, Qt::AlignCenter);

  setAttribute(Qt::WA_OpaquePaintEvent);
  QObject::connect(uiState(), &UIState::uiUpdate, this, &BodyWindow::updateState);
}

void BodyWindow::updateState(const UIState &s) {
  // do updates based on UIState here
}

void BodyWindow::paintEvent(QPaintEvent *event) {
  QPainter p(this);

  // do any painting here
  p.fillRect(rect(), QColor(0, 0, 0, 255));
}
