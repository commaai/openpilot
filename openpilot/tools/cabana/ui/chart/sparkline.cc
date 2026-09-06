#include "tools/cabana/ui/chart/sparkline.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "tools/cabana/ui/util.h"

void Sparkline::update(const cabana::Signal *sig, CanEventIter first, CanEventIter last, int range, ImVec2 sz,
                       double window_end) {
  if (first == last || sz.x <= 0 || sz.y <= 0) {
    render_points_.clear();
    size = {};
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
      double x = can->toSeconds((*it)->mono_time) - window_start;
      // the caller hands over a bit of data from before the window so the oldest samples walk out under
      // the clip rect instead of vanishing at the left edge; they must not move the scale though
      if (x >= 0.0) {
        min_val = std::min(min_val, value);
        max_val = std::max(max_val, value);
      }
      points_.push_back({x, value});
    }
  }
  if (min_val > max_val) {  // nothing inside the window, the lead in is all there is
    for (const auto &p : points_) {
      min_val = std::min(min_val, p.y);
      max_val = std::max(max_val, p.y);
    }
  }

  if (points_.empty()) {
    render_points_.clear();
    size = {};
    return;
  }

  freq_ = points_.size() / std::max(points_.back().x - points_.front().x, 1.0);
  render(sig->color, range, sz, window_end);
}

void Sparkline::render(const CabanaColor &color, int range, ImVec2 sz, double window_end) {
  bool is_flat_line = min_val == max_val;
  if (is_flat_line) {
    min_val -= 1.0;
    max_val += 1.0;
  }

  const double xscale = (sz.x - 1) / (double)range;
  // Leave room for the stroke's antialiasing fringe and the 3x3 endpoint marker
  // at both extrema, including the half-pixel offset applied by ImGui's strokes.
  const double ypadding = std::min(3.0, sz.y / 2.0);
  const double yscale = (sz.y - 2 * ypadding) / (max_val - min_val);
  const double span = points_.back().x - points_.front().x;
  bool draw_individual_points = (span * xscale / points_.size()) > 8.0;

  // transform or downsample the points
  render_points_.reserve(points_.size());
  render_points_.clear();
  if (draw_individual_points) {
    for (const auto &p : points_) {
      render_points_.emplace_back(p.x * xscale, ypadding + (max_val - p.y) * yscale);
    }
  } else if (is_flat_line) {
    double y = sz.y / 2.0;
    render_points_.emplace_back(points_.front().x * xscale, y);
    render_points_.emplace_back(points_.back().x * xscale, y);
  } else {
    double prev_y = points_.front().y;
    render_points_.emplace_back(points_.front().x * xscale, ypadding + (max_val - prev_y) * yscale);
    bool in_flat = false;

    for (size_t i = 1; i < points_.size(); ++i) {
      const auto &p = points_[i];
      double y = p.y;
      if (std::abs(y - prev_y) < 1e-6) {
        in_flat = true;
      } else {
        if (in_flat) render_points_.emplace_back(points_[i - 1].x * xscale, ypadding + (max_val - prev_y) * yscale);
        render_points_.emplace_back(p.x * xscale, ypadding + (max_val - y) * yscale);
        in_flat = false;
      }
      prev_y = y;
    }
    if (in_flat) render_points_.emplace_back(points_.back().x * xscale, ypadding + (max_val - prev_y) * yscale);
  }

  size = sz;
  CabanaColor line_color = color;
  if (!isDarkTheme()) {
    auto [h, s, v] = color.hsv();
    line_color = CabanaColor::fromHsv(h, std::min(1.0f, s * 2.0f), v * 0.7f, color.a / 255.0f);
  }
  color_ = toImU32(line_color);
  draw_individual_points_ = draw_individual_points;
  window_end_ = window_end;
  xscale_ = xscale;
}

void Sparkline::draw(ImDrawList *draw_list, ImVec2 pos, ImU32 color) const {
  if (render_points_.empty()) return;
  if (color == 0) color = color_;

  // update() only runs when a message of this id arrives, so a slow message would hold the sparkline
  // still for many frames and then move it in one step. scroll the rendered polyline by the time that
  // has passed since it was built, which keeps the motion at the frame rate whatever the message rate is
  const float shift = std::clamp((float)((can->currentSec() - window_end_) * xscale_), 0.0f, size.x);

  // physical pixels: the framebuffer is 2x the logical size on hidpi displays
  const float px = 1.0f / std::max(1.0f, ImGui::GetIO().DisplayFramebufferScale.x);
  auto snap = [&](float x) { return std::floor(x / px) * px; };

  // a sample sits at t * xscale + k on screen, where k moves with the clock. snapping k to whole pixels
  // keeps every sample's subpixel phase fixed to its timestamp, so the line looks identical from frame
  // to frame and from update to update and only ever moves by whole pixels; moving it by fractions
  // shifted the antialiasing coverage every frame and the thin peaks sparkled
  ImVec2 offset(pos.x - shift, pos.y);
  const double k = offset.x - (window_end_ * xscale_ - (size.x - 1));
  offset.x += snap(k) - k;
  auto point_at = [&](const ImVec2 &p) { return ImVec2(offset.x + p.x, offset.y + p.y); };

  draw_list->PushClipRect(pos, ImVec2(pos.x + size.x, pos.y + size.y), true);

  // a point is a 3x3 square
  auto draw_point = [&](const ImVec2 &p) { draw_list->AddRectFilled(ImVec2(p.x - 1.5f, p.y - 1.5f), ImVec2(p.x + 1.5f, p.y + 1.5f), color); };

  // Keep ordinary curves joined so short segments don't leave antialiasing seams.
  // Split at turns sharper than 120 degrees: a joined narrow spike folds back on
  // itself and loses its tip. Both paths retain the actual peak and every sample.
  const float thickness = 1.5f;
  for (size_t i = 0; i < render_points_.size(); ++i) {
    const auto &p = render_points_[i];
    draw_list->PathLineTo(point_at(p));
    if (i > 0 && i + 1 < render_points_.size()) {
      const auto &prev = render_points_[i - 1];
      const auto &next = render_points_[i + 1];
      const double ax = p.x - prev.x, ay = p.y - prev.y;
      const double bx = next.x - p.x, by = next.y - p.y;
      const double dot = ax * bx + ay * by;
      if (dot < 0 && 4 * dot * dot > (ax * ax + ay * ay) * (bx * bx + by * by)) {
        draw_list->PathStroke(color, ImDrawFlags_None, thickness);
        draw_list->PathLineTo(point_at(p));
      }
    }
  }
  draw_list->PathStroke(color, ImDrawFlags_None, thickness);
  if (draw_individual_points_) {
    for (const auto &p : render_points_) draw_point(point_at(p));
  } else {
    draw_point(point_at(render_points_.back()));
  }
  draw_list->PopClipRect();
}
