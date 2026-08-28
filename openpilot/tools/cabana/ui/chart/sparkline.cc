#include "tools/cabana/ui/chart/sparkline.h"

#include <algorithm>
#include <cmath>
#include <limits>

void Sparkline::update(const cabana::Signal *sig, CanEventIter first, CanEventIter last, int range, ImVec2 size) {
  if (first == last || size.x <= 0 || size.y <= 0) {
    render_points_.clear();
    this->size = {};
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
      points_.push_back({((*it)->mono_time - start_time) / 1e9, value});
    }
  }

  if (points_.empty()) {
    render_points_.clear();
    this->size = {};
    return;
  }

  freq_ = points_.size() / std::max(points_.back().x - points_.front().x, 1.0);
  render(sig->color, range, size);
}

void Sparkline::render(const CabanaColor &color, int range, ImVec2 size) {
  // Adjust for flat lines
  bool is_flat_line = min_val == max_val;
  if (is_flat_line) {
    min_val -= 1.0;
    max_val += 1.0;
  }

  // Calculate scaling
  const double xscale = (size.x - 1) / (double)range;
  const double yscale = (size.y - 3) / (max_val - min_val);
  bool draw_individual_points = (points_.back().x * xscale / points_.size()) > 8.0;

  // Transform or downsample points
  render_points_.reserve(points_.size());
  render_points_.clear();
  if (draw_individual_points) {
    for (const auto &p : points_) {
      render_points_.emplace_back(p.x * xscale, 1.0 + (max_val - p.y) * yscale);
    }
  } else if (is_flat_line) {
    double y = size.y / 2.0;
    render_points_.emplace_back(0.0, y);
    render_points_.emplace_back(points_.back().x * xscale, y);
  } else {
    double prev_y = points_.front().y;
    render_points_.emplace_back(points_.front().x * xscale, 1.0 + (max_val - prev_y) * yscale);
    bool in_flat = false;

    for (size_t i = 1; i < points_.size(); ++i) {
      const auto &p = points_[i];
      double y = p.y;
      if (std::abs(y - prev_y) < 1e-6) {
        in_flat = true;
      } else {
        if (in_flat) render_points_.emplace_back(points_[i - 1].x * xscale, 1.0 + (max_val - prev_y) * yscale);
        render_points_.emplace_back(p.x * xscale, 1.0 + (max_val - y) * yscale);
        in_flat = false;
      }
      prev_y = y;
    }
    if (in_flat) render_points_.emplace_back(points_.back().x * xscale, 1.0 + (max_val - prev_y) * yscale);
  }

  // Render to pixmap: imgui redraws every frame, so only the polyline and its style are kept (see draw())
  this->size = size;
  color_ = IM_COL32(color.r, color.g, color.b, color.a);
  draw_individual_points_ = draw_individual_points;
}

void Sparkline::draw(ImDrawList *draw_list, ImVec2 pos) const {
  if (render_points_.empty()) return;

  std::vector<ImVec2> pts;
  pts.reserve(render_points_.size());
  for (const auto &p : render_points_) pts.emplace_back(pos.x + p.x, pos.y + p.y);

  // painter.setRenderHint(QPainter::Antialiasing, render_points_.size() <= 500)
  const ImDrawListFlags backup_flags = draw_list->Flags;
  if (render_points_.size() > 500) draw_list->Flags &= ~ImDrawListFlags_AntiAliasedLines;
  draw_list->AddPolyline(pts.data(), (int)pts.size(), color_, ImDrawFlags_None, 1.0f);

  // QPen(color, 3) points
  if (draw_individual_points_) {
    for (const auto &p : pts) draw_list->AddCircleFilled(p, 1.5f, color_);
  } else {
    draw_list->AddCircleFilled(pts.back(), 1.5f, color_);
  }
  draw_list->Flags = backup_flags;
}
