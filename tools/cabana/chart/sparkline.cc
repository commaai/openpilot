#include "tools/cabana/chart/sparkline.h"

#include <limits>
#include <QPainter>

#include "tools/cabana/streams/abstractstream.h"

void Sparkline::update(const MessageId &msg_id, const cabana::Signal *sig, double last_msg_ts, int range, QSize size) {
  const auto &msgs = can->events(msg_id);
  uint64_t ts = (last_msg_ts + can->routeStartTime()) * 1e9;
  uint64_t first_ts = (ts > range * 1e9) ? ts - range * 1e9 : 0;
  auto first = std::lower_bound(msgs.cbegin(), msgs.cend(), first_ts, CompareCanEvent());
  auto last = std::upper_bound(first, msgs.cend(), ts, CompareCanEvent());

  bool update_values = last_ts != last_msg_ts || time_range != range;
  last_ts = last_msg_ts;
  time_range = range;

  if (first != last) {
    if (update_values) {
      values.clear();
      if (values.capacity() < std::distance(first, last)) {
        values.reserve(std::distance(first, last) * 2);
      }
      min_val = std::numeric_limits<double>::max();
      max_val = std::numeric_limits<double>::lowest();
      for (auto it = first; it != last; ++it) {
        const CanEvent *e = *it;
        double value = 0;
        if (sig->getValue(e->dat, e->size, &value)) {
          values.emplace_back((e->mono_time - (*first)->mono_time) / 1e9, value);
          if (min_val > value) min_val = value;
          if (max_val < value) max_val = value;
        }
      }
      if (min_val == max_val) {
        min_val -= 1;
        max_val += 1;
      }
    }
  } else {
    values.clear();
  }

  if (!values.empty()) {
    render(sig->color, size);
  } else {
    pixmap = QPixmap();
    min_val = -1;
    max_val = 1;
  }
}

void Sparkline::render(const QColor &color, QSize size) {
  const double xscale = (size.width() - 1) / (double)time_range;
  const double yscale = (size.height() - 3) / (max_val - min_val);
  points.clear();
  points.reserve(values.capacity());
  for (auto &v : values) {
    points.emplace_back(v.x() * xscale, 1 + std::abs(v.y() - max_val) * yscale);
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
