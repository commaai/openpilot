#include "tools/cabana/chart/sparkline.h"

#include <algorithm>
#include <limits>
#include <QPainter>

void Sparkline::update(const cabana::Signal *sig, CanEventIter first, CanEventIter last, int range, QSize size) {
  if (first == last || size.isEmpty()) {
    pixmap = QPixmap();
    return;
  }

  points_.clear();
  min_val = std::numeric_limits<double>::max();
  max_val = std::numeric_limits<double>::lowest();
  points_.reserve(std::distance(first, last));

  uint64_t start_time = (*first)->mono_time;
  double value = 0.0;
  for (auto it = first; it != last; ++it) {
    if (sig->getValue((*it)->dat, (*it)->size, &value)) {
      min_val = std::min(min_val, value);
      max_val = std::max(max_val, value);
      points_.emplace_back(((*it)->mono_time - start_time) / 1e9, value);
    }
  }

  if (points_.empty()) {
    pixmap = QPixmap();
    return;
  }

  freq_ = points_.size() / std::max(points_.back().x() - points_.front().x(), 1.0);
  render(sig->color, range, size);
}

void Sparkline::render(const QColor &color, int range, QSize size) {
  // Adjust for flat lines
  bool is_flat_line = min_val == max_val;
  if (is_flat_line) {
    min_val -= 1.0;
    max_val += 1.0;
  }

  // Calculate scaling
  const double xscale = (size.width() - 1) / (double)range;
  const double yscale = (size.height() - 3) / (max_val - min_val);
  bool draw_individual_points = (points_.back().x() * xscale / points_.size()) > 8.0;

  // Transform or downsample points
  render_points_.reserve(points_.size());
  render_points_.clear();
  if (draw_individual_points) {
    for (const auto &p : points_) {
      render_points_.emplace_back(p.x() * xscale, 1.0 + (max_val - p.y()) * yscale);
    }
  } else if (is_flat_line) {
    double y = size.height() / 2.0;
    render_points_.emplace_back(0.0, y);
    render_points_.emplace_back(points_.back().x() * xscale, y);
  } else {
    double prev_y = points_.front().y();
    render_points_.emplace_back(points_.front().x() * xscale, 1.0 + (max_val - prev_y) * yscale);
    bool in_flat = false;

    for (size_t i = 1; i < points_.size(); ++i) {
      const auto &p = points_[i];
      double y = p.y();
      if (std::abs(y - prev_y) < 1e-6) {
        in_flat = true;
      } else {
        if (in_flat) render_points_.emplace_back(points_[i - 1].x() * xscale, 1.0 + (max_val - prev_y) * yscale);
        render_points_.emplace_back(p.x() * xscale, 1.0 + (max_val - y) * yscale);
        in_flat = false;
      }
      prev_y = y;
    }
    if (in_flat) render_points_.emplace_back(points_.back().x() * xscale, 1.0 + (max_val - prev_y) * yscale);
  }

  // Render to pixmap
  qreal dpr = qApp->devicePixelRatio();
  const QSize pixmap_size = size * dpr;
  if (pixmap.size() != pixmap_size) {
    pixmap = QPixmap(pixmap_size);
  }
  pixmap.setDevicePixelRatio(dpr);
  pixmap.fill(Qt::transparent);
  QPainter painter(&pixmap);
  painter.setRenderHint(QPainter::Antialiasing, render_points_.size() <= 500);
  painter.setPen(color);
  painter.drawPolyline(render_points_.data(), render_points_.size());

  painter.setPen(QPen(color, 3));
  if (draw_individual_points) {
    painter.drawPoints(render_points_.data(), render_points_.size());
  } else {
    painter.drawPoint(render_points_.back());
  }
}
