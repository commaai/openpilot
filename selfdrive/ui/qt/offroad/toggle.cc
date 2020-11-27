#include "toggle.hpp"

#include <QAbstractButton>
#include <QPropertyAnimation>
#include <QWidget>
#include <QDebug>

Toggle::Toggle(QWidget *parent) : QAbstractButton(parent),
_height(45),
_on(false),
_padding_circle(0),
_anim(new QPropertyAnimation(this, "offset_circle", this))
{
  _radius = 1 + _height / 2;
  _x_circle = _radius;
  _y_circle = _radius;
}


void Toggle::paintEvent(QPaintEvent *e) {
  this->setFixedHeight(_height);
  QPainter p(this);
  p.setPen(Qt::NoPen);
  p.setRenderHint(QPainter::Antialiasing, true);
  // Draw toggle background
  p.setBrush(_on ? Qt::gray : Qt::black);
  p.drawRoundedRect(QRect(0, 0, width(), _height), _radius, _radius);
  // Draw toggle circle
  p.setBrush(_on ? QColor("#ffffff") : QColor("#0000ff"));
  p.drawEllipse(QRectF(_x_circle - _radius, _y_circle - _radius - 1, 2 * _radius, 2 * _radius));
}

void Toggle::mouseReleaseEvent(QMouseEvent *e) {
  if (e->button() & Qt::LeftButton) {
    togglePosition();
    QAbstractButton::mouseReleaseEvent(e);
  }
}
void Toggle::togglePosition(){
  _on = !_on;
  if (_on) {
    _anim->setStartValue(_radius);
    _anim->setEndValue(width() - _radius);
    _anim->setDuration(600);
    _anim->start();

  } else {
    _anim->setStartValue(width() - _radius);
    _anim->setEndValue(_radius);
    _anim->setDuration(600);
    _anim->start();

  }
}

void Toggle::enterEvent(QEvent *e) {
  setCursor(Qt::PointingHandCursor);
  QAbstractButton::enterEvent(e);
}
