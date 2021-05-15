#pragma once

#include <QtWidgets>

inline void configFont(QPainter &p, QString family, int size, const QString &style) {
  QFont f(family);
  f.setPixelSize(size);
  f.setStyleName(style);
  p.setFont(f);
}
