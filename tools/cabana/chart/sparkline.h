#pragma once

#include <algorithm>

#include <QPixmap>
#include <QPointF>
#include <vector>

#include "tools/cabana/dbc/dbcmanager.h"

class Sparkline {
public:
  void update(const MessageId &msg_id, const cabana::Signal *sig, double last_msg_ts, int range, QSize size);
  const QSize size() const { return pixmap.size() / pixmap.devicePixelRatio(); }
  inline double freq() const {
    return values.empty() ? 0 : values.size() / std::max(values.back().x() - values.front().x(), 1.0);
  }

  QPixmap pixmap;
  double min_val = 0;
  double max_val = 0;
  double last_ts = 0;
  int time_range = 0;

private:
  void render(const QColor &color, QSize size);
  std::vector<QPointF> values;
  std::vector<QPointF> points;
};
