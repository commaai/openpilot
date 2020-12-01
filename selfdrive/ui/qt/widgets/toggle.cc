#include "toggle.hpp"

Toggle::Toggle(QWidget *parent) : QAbstractButton(parent),
_height(80),
_height_rect(60),
_on(false),
_anim(new QPropertyAnimation(this, "offset_circle", this))
{
  _radius = _height / 2;
  _x_circle = _radius;
  _y_circle = _radius;
  _y_rect = (_height - _height_rect)/2;
}

void Toggle::paintEvent(QPaintEvent *e) {
  this->setFixedHeight(_height);
  QPainter p(this);
  p.setPen(Qt::NoPen);
  p.setRenderHint(QPainter::Antialiasing, true);

  // Draw toggle background left
  p.setBrush(QColor("#33ab4c"));
  p.drawRoundedRect(QRect(0, _y_rect, _x_circle + _radius, _height_rect), _height_rect/2, _height_rect/2);
  // Draw toggle background right
  p.setBrush(QColor("#0a1a26"));
  p.drawRoundedRect(QRect(_x_circle - _radius, _y_rect, width() -(_x_circle - _radius), _height_rect), _height_rect/2, _height_rect/2);

  // Draw toggle circle
  p.setBrush(QColor("#fafafa"));
  p.drawEllipse(QRectF(_x_circle - _radius, _y_circle - _radius, 2 * _radius, 2 * _radius));
}

void Toggle::mouseReleaseEvent(QMouseEvent *e) {
  if (e->button() & Qt::LeftButton) {
    togglePosition();
    emit stateChanged(_on);
  }
}

void Toggle::togglePosition() {
  _on = !_on;
  const int left = _radius;
  const int right = width() - _radius;
  _anim->setStartValue(_on ? left : right);
  _anim->setEndValue(_on ? right : left);
  _anim->setDuration(120);
  _anim->start();
}

void Toggle::enterEvent(QEvent *e) {
  QAbstractButton::enterEvent(e);
}
