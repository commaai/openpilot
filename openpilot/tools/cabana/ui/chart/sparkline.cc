#include "tools/cabana/ui/chart/sparkline.h"

#include <algorithm>
#include <cmath>
#include <limits>

void Sparkline::update(const cabana::Signal *sig, CanEventIter first, CanEventIter last, int range, ImVec2 sz,
                       double window_end) {
  if (first == last || sz.x <= 0 || sz.y <= 0) {
    render_points_.clear();
    this->size = {};
    return;
  }

  points_.clear();
  min_val = std::numeric_limits<double>::max();
  max_val = std::numeric_limits<double>::lowest();
  points_.reserve(std::distance(first, last));

  // x runs from the start of the time window, not from the first sample in it: the oldest sample drops
  // out at its own rate, so anchoring to it slid the whole curve sideways by the sample spacing on every
  // update and the sparkline jittered left and right instead of scrolling with the clock
  const double window_start = window_end - range;
  double value = 0.0;
  for (auto it = first; it != last; ++it) {
    if (sig->getValue((*it)->dat, (*it)->size, &value)) {
      min_val = std::min(min_val, value);
      max_val = std::max(max_val, value);
      points_.push_back({can->toSeconds((*it)->mono_time) - window_start, value});
    }
  }

  if (points_.empty()) {
    render_points_.clear();
    this->size = {};
    return;
  }

  freq_ = points_.size() / std::max(points_.back().x - points_.front().x, 1.0);
  render(sig->color, range, sz);
}

void Sparkline::render(const CabanaColor &color, int range, ImVec2 sz) {
  bool is_flat_line = min_val == max_val;
  if (is_flat_line) {
    min_val -= 1.0;
    max_val += 1.0;
  }

  const double xscale = (sz.x - 1) / (double)range;
  const double yscale = (sz.y - 3) / (max_val - min_val);
  const double span = points_.back().x - points_.front().x;
  bool draw_individual_points = (span * xscale / points_.size()) > 8.0;

  // transform or downsample the points
  render_points_.reserve(points_.size());
  render_points_.clear();
  if (draw_individual_points) {
    for (const auto &p : points_) {
      render_points_.emplace_back(p.x * xscale, 1.0 + (max_val - p.y) * yscale);
    }
  } else if (is_flat_line) {
    double y = sz.y / 2.0;
    render_points_.emplace_back(points_.front().x * xscale, y);
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

  this->size = sz;
  color_ = IM_COL32(color.r, color.g, color.b, color.a);
  draw_individual_points_ = draw_individual_points;
}

void Sparkline::draw(ImDrawList *draw_list, ImVec2 pos) const {
  if (render_points_.empty()) return;

  std::vector<ImVec2> pts;
  pts.reserve(render_points_.size());
  for (const auto &p : render_points_) pts.emplace_back(pos.x + p.x, pos.y + p.y);

  // an aliased 1 px segment between two columns rounds into one of them and the rounding flips as the
  // window slides, so the thin peaks sparkle; antialiasing spreads it over both and the motion is smooth
  draw_list->AddPolyline(pts.data(), (int)pts.size(), color_, ImDrawFlags_None, 1.0f);

  // a point is a 3x3 square
  auto draw_point = [&](const ImVec2 &p) { draw_list->AddRectFilled(ImVec2(p.x - 1.5f, p.y - 1.5f), ImVec2(p.x + 1.5f, p.y + 1.5f), color_); };
  if (draw_individual_points_) {
    for (const auto &p : pts) draw_point(p);
  } else {
    draw_point(pts.back());
  }
}
