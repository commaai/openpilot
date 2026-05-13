#pragma once

#include <QPixmap>
#include <QPointF>
#include <vector>

#include "tools/cabana/dbc/dbc.h"
#include "tools/cabana/streams/abstractstream.h"

class Sparkline {
public:
  void update(const cabana::Signal *sig, CanEventIter first, CanEventIter last, int range, QSize size);
  inline double freq() const { return freq_; }
  bool isEmpty() const { return pixmap.isNull(); }

  QPixmap pixmap;
  double min_val = 0;
  double max_val = 0;

private:
  void render(const QColor &color, int range, QSize size);

  std::vector<QPointF> points_;
  std::vector<QPointF> render_points_;
  double freq_ = 0;
};
