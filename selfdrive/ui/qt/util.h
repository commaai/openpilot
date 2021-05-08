#pragma once

#include <QtWidgets>

inline void configFont(QPainter &p, QString family, int size, int weight) {
  QFont f(family);
  f.setPixelSize(size);
  f.setWeight(weight);
  p.setFont(f);
}
