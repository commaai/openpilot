/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/widgets/toggle.h"

#include <QPainter>

ToggleSP::ToggleSP(QWidget *parent) : Toggle(parent) {
  _height_rect = 80;
}

void ToggleSP::paintEvent(QPaintEvent *e) {
  this->setFixedHeight(100);
  QPainter p(this);
  p.setPen(Qt::NoPen);
  p.setRenderHint(QPainter::Antialiasing, true);

  // Draw toggle background
  enabled ? green.setRgb(0x1e79e8) : green.setRgb(0x125db8);
  p.setBrush(on ? green : QColor(0x292929));
  p.drawRoundedRect(QRect(0, 10, width(), _height_rect), _height_rect / 2, _height_rect / 2);

  // Draw toggle circle
  p.setBrush(circleColor);
  p.drawEllipse(QRectF(_x_circle - _radius + 6, 16, 68, 68));
}
