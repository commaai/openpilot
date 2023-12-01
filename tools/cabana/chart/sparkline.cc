#include "tools/cabana/chart/sparkline.h"

#include <algorithm>
#include <limits>
#include <QPainter>

#include "tools/cabana/streams/abstractstream.h"

void Sparkline::update(const MessageId &msg_id, const cabana::Signal *sig, double last_msg_ts, int range, QSize size) {
  const auto &msgs = can->events(msg_id);
  uint64_t ts = (last_msg_ts + can->routeStartTime()) * 1e9;
  uint64_t first_ts = (ts > range * 1e9) ? ts - range * 1e9 : 0;
  auto first = std::lower_bound(msgs.cbegin(), msgs.cend(), first_ts, CompareCanEvent());
  auto last = std::upper_bound(first, msgs.cend(), ts, CompareCanEvent());

  if (first != last && !size.isEmpty()) {
    points.clear();
    double value = 0;
    for (auto it = first; it != last; ++it) {
      if (sig->getValue((*it)->dat, (*it)->size, &value)) {
        points.emplace_back(((*it)->mono_time - (*first)->mono_time) / 1e9, value);
      }
    }
    const auto [min, max] = std::minmax_element(points.begin(), points.end(),
                                                [](auto &l, auto &r) { return l.y() < r.y(); });
    min_val = min->y() == max->y() ? min->y() - 1 : min->y();
    max_val = min->y() == max->y() ? max->y() + 1 : max->y();
    freq_ = points.size() / std::max(points.back().x() - points.front().x(), 1.0);
    render(sig->color, range, size);
  } else {
    pixmap = QPixmap();
  }
}

void Sparkline::render(const QColor &color, int range, QSize size) {
  const double xscale = (size.width() - 1) / (double)range;
  const double yscale = (size.height() - 3) / (max_val - min_val);
  for (auto &v : points) {
    v = QPoint(v.x() * xscale, 1 + std::abs(v.y() - max_val) * yscale);
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
