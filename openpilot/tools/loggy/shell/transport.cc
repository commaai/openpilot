#include "tools/loggy/shell/transport.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace loggy {
namespace {

constexpr double kEpsilon = 1.0e-9;

bool finite(double value) {
  return std::isfinite(value);
}

double clamp_fraction(double fraction) {
  if (!finite(fraction)) return 0.0;
  return std::clamp(fraction, 0.0, 1.0);
}

TimeRange intersect_ranges(TimeRange a, TimeRange b) {
  a = normalize_time_range(a);
  b = normalize_time_range(b);
  return {std::max(a.start_, b.start_), std::min(a.end, b.end)};
}

int span_priority(TimelineSpanKind kind) {
  switch (kind) {
    case TimelineSpanKind::AlertCritical:
      return 4;
    case TimelineSpanKind::AlertWarning:
      return 3;
    case TimelineSpanKind::AlertInfo:
      return 2;
    case TimelineSpanKind::Engaged:
      return 1;
    case TimelineSpanKind::None:
    default:
      return 0;
  }
}

bool span_less(const TimelineSpan &a, const TimelineSpan &b) {
  if (std::abs(a.start_time - b.start_time) > kEpsilon) return a.start_time < b.start_time;
  if (std::abs(a.end_time - b.end_time) > kEpsilon) return a.end_time < b.end_time;
  return span_priority(a.kind) < span_priority(b.kind);
}

}  // namespace

bool TimeRange::valid() const {
  return finite(start_) && finite(end) && end >= start_;
}

double TimeRange::span() const {
  return valid() ? end - start_ : 0.0;
}

TimeRange normalize_time_range(TimeRange range, double fallback_span) {
  if (!finite(fallback_span) || fallback_span < 0.0) fallback_span = 1.0;

  const bool start_ok = finite(range.start_);
  const bool end_ok = finite(range.end);
  if (!start_ok && !end_ok) return {0.0, fallback_span};
  if (!start_ok) range.start_ = range.end;
  if (!end_ok) range.end = range.start_ + fallback_span;
  if (range.end < range.start_) std::swap(range.start_, range.end);
  return range;
}

TimeRange clamp_time_range(TimeRange range, TimeRange bounds, double min_span) {
  bounds = normalize_time_range(bounds);
  range = normalize_time_range(range, bounds.span());
  if (!finite(min_span) || min_span < 0.0) min_span = 0.0;

  const double bound_span = bounds.span();
  if (bound_span <= 0.0) return bounds;
  min_span = std::min(min_span, bound_span);

  double width = range.span();
  if (width < min_span) {
    const double center = (range.start_ + range.end) * 0.5;
    range.start_ = center - min_span * 0.5;
    range.end = center + min_span * 0.5;
    width = min_span;
  }

  if (width >= bound_span) return bounds;
  if (range.start_ < bounds.start_) {
    range.end += bounds.start_ - range.start_;
    range.start_ = bounds.start_;
  }
  if (range.end > bounds.end) {
    range.start_ -= range.end - bounds.end;
    range.end = bounds.end;
  }
  if (range.start_ < bounds.start_) range.start_ = bounds.start_;
  if (range.end > bounds.end) range.end = bounds.end;
  if (range.end < range.start_) range.end = range.start_;
  return range;
}

double clamp_time_to_range(double time, TimeRange range) {
  range = normalize_time_range(range);
  if (!finite(time)) return range.start_;
  if (range.span() <= 0.0) return range.start_;
  return std::clamp(time, range.start_, range.end);
}

double fraction_to_time(double fraction, TimeRange range) {
  range = normalize_time_range(range);
  return range.start_ + clamp_fraction(fraction) * range.span();
}

double time_to_fraction(double time, TimeRange range) {
  range = normalize_time_range(range);
  const double span = range.span();
  if (span <= 0.0) return 0.0;
  return (clamp_time_to_range(time, range) - range.start_) / span;
}

PlaybackClock::PlaybackClock() {
  init(TimeRange{0.0, 1.0});
}

void PlaybackClock::init(TimeRange route_range) {
  route_range_ = normalize_time_range(route_range);
  tracker_time_ = route_range_.start_;
  rate_ = 1.0;
  playing_ = false;
  loop_ = false;
}

void PlaybackClock::set_route_range(TimeRange route_range) {
  route_range_ = normalize_time_range(route_range);
  tracker_time_ = clamp_time_to_range(tracker_time_, route_range_);
}

void PlaybackClock::seek(double time) {
  tracker_time_ = clamp_time_to_range(time, route_range_);
}

bool PlaybackClock::toggle_playing() {
  playing_ = !playing_;
  return playing_;
}

void PlaybackClock::set_playing(bool playing) {
  playing_ = playing;
}

void PlaybackClock::set_rate(double rate) {
  rate_ = finite(rate) ? std::max(0.0, rate) : 1.0;
}

void PlaybackClock::set_loop(bool loop) {
  loop_ = loop;
}

double PlaybackClock::step_forward() {
  return step(1);
}

double PlaybackClock::step_backward() {
  return step(-1);
}

double PlaybackClock::step(int direction) {
  if (direction == 0) return tracker_time_;
  const double sign = direction < 0 ? -1.0 : 1.0;
  seek(tracker_time_ + sign * kDefaultPlaybackStepSeconds);
  return tracker_time_;
}

void PlaybackClock::advance(double delta_seconds) {
  if (!playing_ || !finite(delta_seconds) || delta_seconds <= 0.0 || rate_ <= 0.0 || route_range_.span() <= 0.0) {
    return;
  }

  double base_time = tracker_time_;
  if (loop_ && base_time >= route_range_.end) base_time = route_range_.start_;

  double next_time = base_time + delta_seconds * rate_;
  if (next_time >= route_range_.end) {
    if (loop_) {
      next_time = route_range_.start_;
    } else {
      next_time = route_range_.end;
      playing_ = false;
    }
  }
  tracker_time_ = clamp_time_to_range(next_time, route_range_);
}

SharedViewRange::SharedViewRange() : SharedViewRange(TimeRange{0.0, 1.0}) {}

SharedViewRange::SharedViewRange(TimeRange route_range) {
  set_route_range(route_range);
  reset_to_route();
}

void SharedViewRange::set_route_range(TimeRange route_range) {
  route_range_ = normalize_time_range(route_range);
  range_ = clamp_time_range(range_, route_range_);
}

void SharedViewRange::set_range(TimeRange range) {
  range_ = clamp_time_range(range, route_range_);
}

void SharedViewRange::reset_to_route() {
  range_ = route_range_;
}

TimelineSpanKind timeline_kind_for_alert(AlertLevel level) {
  switch (level) {
    case AlertLevel::Critical:
      return TimelineSpanKind::AlertCritical;
    case AlertLevel::Warning:
      return TimelineSpanKind::AlertWarning;
    case AlertLevel::Info:
    default:
      return TimelineSpanKind::AlertInfo;
  }
}

TimelineColor timeline_span_color(TimelineSpanKind kind, uint8_t alpha) {
  switch (kind) {
    case TimelineSpanKind::Engaged:
      return {0, 163, 108, alpha};
    case TimelineSpanKind::AlertInfo:
      return {255, 195, 0, alpha};
    case TimelineSpanKind::AlertWarning:
    case TimelineSpanKind::AlertCritical:
      return {199, 0, 57, alpha};
    case TimelineSpanKind::None:
    default:
      return {111, 143, 175, alpha};
  }
}

const char *timeline_span_label(TimelineSpanKind kind) {
  switch (kind) {
    case TimelineSpanKind::Engaged:
      return "engaged";
    case TimelineSpanKind::AlertInfo:
      return "alert info";
    case TimelineSpanKind::AlertWarning:
      return "alert warning";
    case TimelineSpanKind::AlertCritical:
      return "alert critical";
    case TimelineSpanKind::None:
    default:
      return "disengaged";
  }
}

TimelineModel::TimelineModel() : TimelineModel(TimeRange{0.0, 1.0}) {}

TimelineModel::TimelineModel(TimeRange route_range) {
  set_route_range(route_range);
}

void TimelineModel::set_route_range(TimeRange route_range) {
  route_range_ = normalize_time_range(route_range);
  clamp_stored_spans();
}

void TimelineModel::clear_spans() {
  spans_.clear();
}

void TimelineModel::set_spans(std::vector<TimelineSpan> spans) {
  spans_ = std::move(spans);
  clamp_stored_spans();
}

void TimelineModel::set_spans(const std::vector<EngagementSpan> &engagements, const std::vector<AlertSpan> &alerts) {
  std::vector<TimelineSpan> spans;
  spans.reserve(engagements.size() + alerts.size());
  for (const EngagementSpan &span : engagements) {
    spans.push_back({span.start_time, span.end_time, TimelineSpanKind::Engaged});
  }
  for (const AlertSpan &span : alerts) {
    spans.push_back({span.start_time, span.end_time, timeline_kind_for_alert(span.level)});
  }
  set_spans(std::move(spans));
}

double TimelineModel::time_from_fraction(double fraction) const {
  return fraction_to_time(fraction, route_range_);
}

double TimelineModel::fraction_from_time(double time) const {
  return time_to_fraction(time, route_range_);
}

double TimelineModel::time_from_pixel(double pixel_x, double pixel_width) const {
  if (!finite(pixel_width) || pixel_width <= 0.0) return route_range_.start_;
  return time_from_fraction(pixel_x / pixel_width);
}

double TimelineModel::pixel_from_time(double time, double pixel_width) const {
  if (!finite(pixel_width) || pixel_width <= 0.0) return 0.0;
  return fraction_from_time(time) * pixel_width;
}

TimelineSpanKind TimelineModel::kind_at_time(double time) const {
  if (!finite(time)) return TimelineSpanKind::None;
  TimelineSpanKind best = TimelineSpanKind::None;
  int best_priority = 0;
  for (const TimelineSpan &span : spans_) {
    if (time + kEpsilon < span.start_time || time - kEpsilon > span.end_time) continue;
    const int priority = span_priority(span.kind);
    if (priority >= best_priority) {
      best = span.kind;
      best_priority = priority;
    }
  }
  return best;
}

std::vector<TimelineRenderSpan> TimelineModel::render_spans() const {
  return render_spans(route_range_);
}

std::vector<TimelineRenderSpan> TimelineModel::render_spans(TimeRange display_range) const {
  display_range = clamp_time_range(display_range, route_range_, 0.0);
  std::vector<TimelineRenderSpan> rendered;
  if (display_range.span() <= 0.0) return rendered;

  rendered.reserve(spans_.size());
  for (const TimelineSpan &span : spans_) {
    const TimeRange visible = intersect_ranges({span.start_time, span.end_time}, display_range);
    if (visible.end <= visible.start_) continue;
    rendered.push_back({
      time_to_fraction(visible.start_, display_range),
      time_to_fraction(visible.end, display_range),
      span.kind,
      timeline_span_color(span.kind),
    });
  }
  return rendered;
}

void TimelineModel::clamp_stored_spans() {
  std::vector<TimelineSpan> clean;
  clean.reserve(spans_.size());
  for (TimelineSpan span : spans_) {
    if (span.kind == TimelineSpanKind::None || !finite(span.start_time) || !finite(span.end_time)) continue;
    if (span.end_time < span.start_time) std::swap(span.start_time, span.end_time);
    TimeRange visible = intersect_ranges({span.start_time, span.end_time}, route_range_);
    if (visible.end <= visible.start_) continue;
    clean.push_back({visible.start_, visible.end, span.kind});
  }

  std::sort(clean.begin(), clean.end(), span_less);
  spans_.clear();
  spans_.reserve(clean.size());
  for (const TimelineSpan &span : clean) {
    if (!spans_.empty() && spans_.back().kind == span.kind && span.start_time <= spans_.back().end_time + kEpsilon) {
      spans_.back().end_time = std::max(spans_.back().end_time, span.end_time);
      continue;
    }
    spans_.push_back(span);
  }
}

}  // namespace loggy
