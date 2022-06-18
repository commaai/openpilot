#include "selfdrive/ui/qt/widgets/toggle.h"

#include <QPainter>

Toggle::Toggle(QWidget *parent) : QAbstractButton(parent),
_height(80),
_height_rect(60),
on(false),
_anim(new QPropertyAnimation(this, "offset_circle", this))
{
  _radius = _height / 2;
  _x_circle = _radius;
  _y_circle = _radius;
  _y_rect = (_height - _height_rect)/2;
  circleColor = QColor(0xffffff); // placeholder
  green = QColor(0xffffff); // placeholder
  setEnabled(true);
}

void Toggle::paintEvent(QPaintEvent *e) {
  this->setFixedHeight(_height);
  QPainter p(this);
  p.setPen(Qt::NoPen);
  p.setRenderHint(QPainter::Antialiasing, true);

  // Draw toggle background left
  p.setBrush(green);
  p.drawRoundedRect(QRect(0, _y_rect, _x_circle + _radius, _height_rect), _height_rect/2, _height_rect/2);

  // Draw toggle background right
  p.setBrush(QColor(0xC92231));
  p.drawRoundedRect(QRect(_x_circle - _radius, _y_rect, width() - (_x_circle - _radius), _height_rect), _height_rect/2, _height_rect/2);

  // Draw toggle circle
  p.setBrush(circleColor);
  p.drawEllipse(QRectF(_x_circle - _radius, _y_circle - _radius, 2 * _radius, 2 * _radius));
}

void Toggle::mouseReleaseEvent(QMouseEvent *e) {
  if (!enabled) {
    return;
  }
  const int left = _radius;
  const int right = width() - _radius;
  if ((_x_circle != left && _x_circle != right) || !this->rect().contains(e->localPos().toPoint())) {
    // If mouse release isn't in rect or animation is running, don't parse touch events
    return;
  }
  if (e->button() & Qt::LeftButton) {
    togglePosition();
    emit stateChanged(on);
  }
}

void Toggle::togglePosition() {
  on = !on;
  const int left = _radius;
  const int right = width() - _radius;
  _anim->setStartValue(on ? left + immediateOffset : right - immediateOffset);
  _anim->setEndValue(on ? right : left);
  _anim->setDuration(animation_duration);
  _anim->start();
  repaint();
}

void Toggle::enterEvent(QEvent *e) {
  QAbstractButton::enterEvent(e);
}

bool Toggle::getEnabled() {
  return enabled;
}

void Toggle::setEnabled(bool value) {
  enabled = value;
  if (value) {
    circleColor.setRgb(0xfafafa);
    green.setRgb(0x33ab4c);
  } else {
    circleColor.setRgb(0x888888);
    green.setRgb(0x227722);
  }
}
