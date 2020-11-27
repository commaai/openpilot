#include "toggle.hpp"

#include <QAbstractButton>
#include <QPropertyAnimation>
#include <QWidget>
#include <QDebug>
#include "common/params.h" 

Toggle::Toggle(QWidget *parent) : QAbstractButton(parent),
_height(60),
_height_rect(45),
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

void Toggle::togglePosition(){
  _on = !_on;
  if (_on) {
    _anim->setStartValue(_radius);
    _anim->setEndValue(width() - _radius);
    _anim->setDuration(120);
    _anim->start();
  } else {
    _anim->setStartValue(width() - _radius);
    _anim->setEndValue(_radius);
    _anim->setDuration(120);
    _anim->start();
  }
}

void Toggle::enterEvent(QEvent *e) {
  setCursor(Qt::PointingHandCursor);
  QAbstractButton::enterEvent(e);
}
