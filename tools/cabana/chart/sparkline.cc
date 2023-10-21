#include "tools/cabana/chart/sparkline.h"

#include <algorithm>
#include <limits>
#include <QPainter>

#include "tools/cabana/streams/abstractstream.h"

void Sparkline::update(const MessageId &msg_id, const cabana::Signal *sig, uint64_t last_ts, int time_range, QSize size) {
  const auto &msgs = can->events(msg_id);
  uint64_t first_ts = last_ts - std::min<uint64_t>(last_ts, time_range * 1e9);
  auto first = std::lower_bound(msgs.cbegin(), msgs.cend(), first_ts, CompareCanEvent());
  auto last = std::upper_bound(first, msgs.cend(), last_ts, CompareCanEvent());

  freq_ = 0;
  points.clear();
  if (first != last) {
    if (points.capacity() < std::distance(first, last)) {
      points.reserve(std::distance(first, last) * 2);
    }
    min_val = std::numeric_limits<double>::max();
    max_val = std::numeric_limits<double>::lowest();
    double value = 0;
    for (auto it = first; it != last; ++it) {
      const CanEvent *e = *it;
      if (sig->getValue(e->dat, e->size, &value)) {
        points.emplace_back((e->mono_time - (*first)->mono_time) / 1e9, value);
        if (min_val > value) min_val = value;
        if (max_val < value) max_val = value;
      }
    }
    freq_ = points.size() / std::max(points.back().x() - points.front().x(), 1.0);
    if (min_val == max_val) {
      min_val -= 1;
      max_val += 1;
    }

    if (size.width() > 0 && size.height() > 0) {
      render(sig->color, time_range, size);
    } else {
      pixmap = QPixmap();
    }
  } else {
    pixmap = QPixmap();
    min_val = -1;
    max_val = 1;
  }
}

void Sparkline::render(const QColor &color, int time_range, QSize size) {
  const double xscale = (size.width() - 1) / (double)time_range;
  const double yscale = (size.height() - 3) / (max_val - min_val);
  for (auto &v : points) {
    v = QPointF(v.x() * xscale, 1 + std::abs(v.y() - max_val) * yscale);
  }

  qreal dpr = qApp->devicePixelRatio();
  size *= dpr;
  if (size != pixmap.size()) {
    pixmap = QPixmap(size);
  }
  pixmap.setDevicePixelRatio(dpr);
  pixmap.fill(Qt::transparent);

  QPainter painter(&pixmap);
  painter.setRenderHint(QPainter::Antialiasing, points.size() < 500);
  painter.setPen(color);
  painter.drawPolyline(points.data(), points.size());
  painter.setPen(QPen(color, 3));
  if ((points.back().x() - points.front().x()) / points.size() > 8) {
    painter.drawPoints(points.data(), points.size());
  } else {
    painter.drawPoint(points.back());
  }
}
