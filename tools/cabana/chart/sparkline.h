#pragma once

#include <QPixmap>
#include <QPointF>
#include <vector>

#include "tools/cabana/dbc/dbcmanager.h"

class Sparkline {
public:
  void update(const MessageId &msg_id, const cabana::Signal *sig, uint64_t last_ts, int time_range, QSize size);
  inline double freq() const { return freq_; }

  QPixmap pixmap;
  double min_val = 0;
  double max_val = 0;

private:
  void render(const QColor &color, int time_range, QSize size);

  std::vector<QPointF> points;
  double freq_ = 0;
};
