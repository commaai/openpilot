#include "clickablelabel.hpp"

#include <QDebug>
#include <QEvent>

ClickableLabel::ClickableLabel(QWidget *parent) : QLabel(parent) {
  setStyleSheet(R"(color: #8a8a8a;)");
}

void ClickableLabel::enterEvent(QEvent *e) {
  setCursor(Qt::PointingHandCursor);
  setStyleSheet(R"(color: #ffffff;)");
}

void ClickableLabel::leaveEvent(QEvent *e) {
  setCursor(Qt::ArrowCursor);
  setStyleSheet(R"(color: #8a8a8a;)");
}

void ClickableLabel::mousePressEvent(QMouseEvent *e) {
  // TODO: Toggle righthand layouts here.
  // Either send a signal or pass in the layout to this class.
}
