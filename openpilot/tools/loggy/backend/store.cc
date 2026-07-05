#include "tools/loggy/backend/store.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <utility>

namespace loggy {
namespace {

constexpr double EPS = 1e-9;

bool eventLess(const CanEvent &a, const CanEvent &b) {
  if (std::abs(a.mono_time - b.mono_time) > EPS) return a.mono_time < b.mono_time;
  return a.bus_time < b.bus_time;
}

TimeRange orderedRange(double t0, double t1) {
  if (t1 < t0) std::swap(t0, t1);
  return {t0, t1};
}

TimeRange pointRange(const std::vector<SeriesPoint> &points) {
  if (points.empty()) return {};
  auto [lo, hi] = std::minmax_element(points.begin(), points.end(), [](const auto &a, const auto &b) {
    return a.t < b.t;
  });
  return {lo->t, hi->t};
}

TimeRange eventRange(const std::vector<CanEvent> &events) {
  if (events.empty()) return {};
  auto [lo, hi] = std::minmax_element(events.begin(), events.end(), [](const auto &a, const auto &b) {
    return a.mono_time < b.mono_time;
  });
  return {lo->mono_time, hi->mono_time};
}

CoverageInfo coverageFor(TimeRange requested, std::vector<TimeRange> ranges) {
  CoverageInfo info;
  info.requested = requested;
  if (!requested.valid()) {
    info.complete = true;
    return info;
  }

  std::vector<TimeRange> clipped;
  clipped.reserve(ranges.size());
  for (const auto &range : ranges) {
    const TimeRange clipped_range = intersection(requested, range);
    if (clipped_range.valid()) clipped.push_back(clipped_range);
  }
  info.ranges = normalizeRanges(std::move(clipped));
  for (const auto &range : info.ranges) {
    info.covered_seconds += range.span();
  }
  info.complete = info.covered_seconds + EPS >= requested.span();
  return info;
}

std::vector<SeriesPoint> decimate(std::vector<SeriesPoint> points, size_t max_points) {
  if (max_points == 0) return {};
  if (points.size() <= max_points) return points;
  if (max_points == 1) return {points.front()};

  std::vector<SeriesPoint> out;
  out.reserve(max_points);
  const size_t last = points.size() - 1;
  const size_t denom = max_points - 1;
  for (size_t i = 0; i < max_points; ++i) {
    out.push_back(points[(i * last) / denom]);
  }
  return out;
}

void mergeEvents(std::vector<CanEvent> *dst, std::vector<CanEvent> incoming) {
  if (incoming.empty()) return;
  if (!std::is_sorted(incoming.begin(), incoming.end(), eventLess)) {
    std::stable_sort(incoming.begin(), incoming.end(), eventLess);
  }
  if (dst->empty()) {
    *dst = std::move(incoming);
    return;
  }

  std::vector<CanEvent> merged;
  merged.reserve(dst->size() + incoming.size());
  std::merge(dst->begin(), dst->end(), incoming.begin(), incoming.end(), std::back_inserter(merged), eventLess);
  *dst = std::move(merged);
}

}  // namespace

bool intersects(TimeRange a, TimeRange b) {
  return a.valid() && b.valid() && a.start < b.end && b.start < a.end;
}

TimeRange intersection(TimeRange a, TimeRange b) {
  if (!intersects(a, b)) return {};
  return {std::max(a.start, b.start), std::min(a.end, b.end)};
}

double distance(TimeRange a, TimeRange b) {
  if (!a.valid() || !b.valid()) return std::numeric_limits<double>::infinity();
  if (intersects(a, b)) return 0.0;
  return a.end <= b.start ? b.start - a.end : a.start - b.end;
}

double distanceToPoint(TimeRange range, double point) {
  if (!range.valid()) return std::numeric_limits<double>::infinity();
  if (point >= range.start && point < range.end) return 0.0;
  return point < range.start ? range.start - point : point - range.end;
}

std::vector<TimeRange> normalizeRanges(std::vector<TimeRange> ranges) {
  ranges.erase(std::remove_if(ranges.begin(), ranges.end(), [](const auto &range) {
    return !range.valid();
  }), ranges.end());
  std::sort(ranges.begin(), ranges.end(), [](const auto &a, const auto &b) {
    if (std::abs(a.start - b.start) > EPS) return a.start < b.start;
    return a.end < b.end;
  });

  std::vector<TimeRange> out;
  out.reserve(ranges.size());
  for (const auto &range : ranges) {
    if (out.empty() || range.start > out.back().end + EPS) {
      out.push_back(range);
    } else {
      out.back().end = std::max(out.back().end, range.end);
    }
  }
  return out;
}

void Store::stage(StoreBatch batch) {
  std::lock_guard lock(staged_mutex_);
  staged_batches_.push_back(std::move(batch));
}

DrainResult Store::beginFrame() {
  std::vector<StoreBatch> batches;
  {
    std::lock_guard lock(staged_mutex_);
    batches.swap(staged_batches_);
  }

  DrainResult result;
  result.batches = batches.size();
  std::vector<std::string> touched_series;
  std::vector<MessageId> touched_can;

  for (auto &batch : batches) {
    std::vector<TimeRange> batch_coverage = batch.coverage;
    for (auto &chunk : batch.series) {
      if (chunk.path.empty()) continue;
      if (!chunk.range.valid()) chunk.range = pointRange(chunk.points);
      if (batch_coverage.empty() && chunk.range.valid()) batch_coverage.push_back(chunk.range);
      if (!std::is_sorted(chunk.points.begin(), chunk.points.end(), [](const auto &a, const auto &b) { return a.t < b.t; })) {
        std::stable_sort(chunk.points.begin(), chunk.points.end(), [](const auto &a, const auto &b) { return a.t < b.t; });
      }

      result.series_points += chunk.points.size();
      ++result.series_chunks;
      touched_series.push_back(chunk.path);
      series_[chunk.path].chunks.push_back(std::move(chunk));
    }

    for (auto &chunk : batch.can_events) {
      if (!chunk.range.valid()) chunk.range = eventRange(chunk.events);
      if (batch_coverage.empty() && chunk.range.valid()) batch_coverage.push_back(chunk.range);

      result.can_events += chunk.events.size();
      ++result.can_chunks;
      touched_can.push_back(chunk.id);
      auto &state = can_events_[chunk.id];
      if (chunk.range.valid()) state.coverage.push_back(chunk.range);
      mergeEvents(&state.events, std::move(chunk.events));
    }

    coverage_.insert(coverage_.end(), batch_coverage.begin(), batch_coverage.end());
  }

  std::sort(touched_series.begin(), touched_series.end());
  touched_series.erase(std::unique(touched_series.begin(), touched_series.end()), touched_series.end());
  for (const auto &path : touched_series) {
    auto &chunks = series_[path].chunks;
    std::stable_sort(chunks.begin(), chunks.end(), [](const auto &a, const auto &b) {
      if (std::abs(a.range.start - b.range.start) > EPS) return a.range.start < b.range.start;
      return a.segment < b.segment;
    });
  }

  std::sort(touched_can.begin(), touched_can.end());
  touched_can.erase(std::unique(touched_can.begin(), touched_can.end()), touched_can.end());
  for (const auto &id : touched_can) {
    auto &state = can_events_[id];
    state.coverage = normalizeRanges(std::move(state.coverage));
  }
  coverage_ = normalizeRanges(std::move(coverage_));

  return result;
}

void Store::clear() {
  {
    std::lock_guard lock(staged_mutex_);
    staged_batches_.clear();
  }
  series_.clear();
  can_events_.clear();
  coverage_.clear();
}

SeriesView Store::series(std::string_view path, double t0, double t1, size_t max_points) const {
  SeriesView view;
  view.path = std::string(path);
  view.requested = orderedRange(t0, t1);
  view.coverage.requested = view.requested;

  const auto it = series_.find(view.path);
  if (it == series_.end()) return view;

  std::vector<TimeRange> ranges;
  std::vector<SeriesPoint> points;
  for (const auto &chunk : it->second.chunks) {
    if (chunk.range.valid()) {
      if (!intersects(chunk.range, view.requested)) continue;
      ranges.push_back(chunk.range);
    }
    for (const auto &point : chunk.points) {
      if (point.t >= view.requested.start && point.t <= view.requested.end) points.push_back(point);
    }
  }

  view.total_points = points.size();
  view.decimated = points.size() > max_points;
  view.points = decimate(std::move(points), max_points);
  view.coverage = coverageFor(view.requested, std::move(ranges));
  return view;
}

CanEventView Store::canEvents(const MessageId &id, TimeRange range) const {
  CanEventView view;
  view.id = id;
  view.requested = orderedRange(range.start, range.end);
  view.coverage.requested = view.requested;

  const auto it = can_events_.find(id);
  if (it == can_events_.end()) return view;

  const auto &events = it->second.events;
  auto first = std::lower_bound(events.begin(), events.end(), view.requested.start,
                                [](const CanEvent &event, double t) { return event.mono_time < t; });
  for (auto e = first; e != events.end() && e->mono_time <= view.requested.end; ++e) {
    view.events.push_back(*e);
  }
  view.coverage = coverageFor(view.requested, it->second.coverage);
  return view;
}

CanSummaryView Store::canEventSummary(const MessageId &id, TimeRange range) const {
  CanSummaryView view;
  view.id = id;
  view.requested = orderedRange(range.start, range.end);
  view.coverage.requested = view.requested;

  const auto it = can_events_.find(id);
  if (it == can_events_.end()) return view;

  const auto &events = it->second.events;
  auto first = std::lower_bound(events.begin(), events.end(), view.requested.start,
                                [](const CanEvent &event, double t) { return event.mono_time < t; });
  auto last = std::upper_bound(first, events.end(), view.requested.end,
                               [](double t, const CanEvent &event) { return t < event.mono_time; });
  view.count = static_cast<size_t>(std::distance(first, last));
  if (first != last) {
    view.first_time = first->mono_time;
    const auto latest = std::prev(last);
    view.last_time = latest->mono_time;
    view.latest_data = latest->data;
  }
  view.coverage = coverageFor(view.requested, it->second.coverage);
  return view;
}

size_t Store::stagedBatchCount() const {
  std::lock_guard lock(staged_mutex_);
  return staged_batches_.size();
}

std::vector<std::string> Store::seriesPaths() const {
  std::vector<std::string> paths;
  paths.reserve(series_.size());
  for (const auto &[path, _] : series_) {
    paths.push_back(path);
  }
  std::sort(paths.begin(), paths.end());
  return paths;
}

std::vector<MessageId> Store::canMessageIds() const {
  std::vector<MessageId> ids;
  ids.reserve(can_events_.size());
  for (const auto &[id, _] : can_events_) {
    ids.push_back(id);
  }
  std::sort(ids.begin(), ids.end());
  return ids;
}

}  // namespace loggy
