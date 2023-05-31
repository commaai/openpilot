#pragma once

#include <QPixmap>
#include <QPointF>
#include <vector>

#include "tools/cabana/dbc/dbcmanager.h"

class Sparkline {
public:
  void update(const MessageId &msg_id, const cabana::Signal *sig, double last_msg_ts, int range, QSize size);
  const QSize size() const { return pixmap.size() / pixmap.devicePixelRatio(); }

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
