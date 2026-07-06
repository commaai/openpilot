#include "tools/loggy/backend/store.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <string_view>
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

bool canReplaceSeriesPath(std::string_view path) {
  return path.rfind("/computed/", 0) == 0;
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
  info.ranges = normalize_ranges(std::move(clipped));
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

std::pair<std::vector<TimeRange>, size_t> trimRangesBefore(std::vector<TimeRange> ranges, double cutoff_time) {
  std::vector<TimeRange> kept;
  kept.reserve(ranges.size());
  size_t removed_ranges = 0;
  for (TimeRange range : ranges) {
    if (!range.valid() || range.end < cutoff_time) {
      ++removed_ranges;
      continue;
    }
    if (range.start_ < cutoff_time) range.start_ = cutoff_time;
    kept.push_back(range);
  }
  return std::pair<std::vector<TimeRange>, size_t>(normalize_ranges(std::move(kept)), removed_ranges);
}

}  // namespace

bool intersects(TimeRange a, TimeRange b) {
  return a.valid() && b.valid() && a.start_ < b.end && b.start_ < a.end;
}

TimeRange intersection(TimeRange a, TimeRange b) {
  if (!intersects(a, b)) return {};
  return {std::max(a.start_, b.start_), std::min(a.end, b.end)};
}

double distance(TimeRange a, TimeRange b) {
  if (!a.valid() || !b.valid()) return std::numeric_limits<double>::infinity();
  if (intersects(a, b)) return 0.0;
  return a.end <= b.start_ ? b.start_ - a.end : a.start_ - b.end;
}

double distance_to_point(TimeRange range, double point) {
  if (!range.valid()) return std::numeric_limits<double>::infinity();
  if (point >= range.start_ && point < range.end) return 0.0;
  return point < range.start_ ? range.start_ - point : point - range.end;
}

std::vector<TimeRange> normalize_ranges(std::vector<TimeRange> ranges) {
  ranges.erase(std::remove_if(ranges.begin(), ranges.end(), [](const auto &range) {
    return !range.valid();
  }), ranges.end());
  std::sort(ranges.begin(), ranges.end(), [](const auto &a, const auto &b) {
    if (std::abs(a.start_ - b.start_) > EPS) return a.start_ < b.start_;
    return a.end < b.end;
  });

  std::vector<TimeRange> out;
  out.reserve(ranges.size());
  for (const auto &range : ranges) {
    if (out.empty() || range.start_ > out.back().end + EPS) {
      out.push_back(range);
    } else {
      out.back().end = std::max(out.back().end, range.end);
    }
  }
  return out;
}

void Store::stage(StoreBatch batch) {
  // Protects staged_batches_ so producers can enqueue while begin_frame drains a previous frame.
  std::lock_guard lock(staged_mutex_);
  staged_batches_.push_back(std::move(batch));
}

DrainResult Store::begin_frame() {
  std::vector<StoreBatch> batches;
  {
    // Move batches under lock so each batch is drained exactly once per frame.
    std::lock_guard lock(staged_mutex_);
    batches.swap(staged_batches_);
  }

  DrainResult result;
  result.batches = batches.size();
  const bool touched_store = !batches.empty();
  std::vector<std::string> touched_series;
  std::vector<MessageId> touched_can;

  for (auto &batch : batches) {
    std::vector<TimeRange> batch_coverage = batch.coverage;
    for (const std::string &path : batch.replace_series_paths) {
      if (!canReplaceSeriesPath(path)) continue;
      series_.erase(path);
      touched_series.push_back(path);
    }

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
  result.touched_series_paths = touched_series;
  for (const auto &path : touched_series) {
    if (series_.find(path) == series_.end()) continue;
    auto &chunks = series_[path].chunks;
    std::stable_sort(chunks.begin(), chunks.end(), [](const auto &a, const auto &b) {
      if (std::abs(a.range.start_ - b.range.start_) > EPS) return a.range.start_ < b.range.start_;
      return a.segment < b.segment;
    });
  }

  std::sort(touched_can.begin(), touched_can.end());
  touched_can.erase(std::unique(touched_can.begin(), touched_can.end()), touched_can.end());
  for (const auto &id : touched_can) {
    auto &state = can_events_[id];
    state.coverage = normalize_ranges(std::move(state.coverage));
  }
  coverage_ = normalize_ranges(std::move(coverage_));
  // Bump generation once so readers that cache generation_ see a single coherent store mutation.
  if (touched_store) ++generation_;

  return result;
}

StoreTrimResult Store::trim_before(double cutoff_time) {
  StoreTrimResult result;
  result.cutoff_time = cutoff_time;
  if (!std::isfinite(cutoff_time)) return result;

  std::vector<std::string> empty_series_paths;
  // Trim points and keep each series' point vectors/references internally consistent per iteration.
  for (auto &[path, state] : series_) {
    std::vector<SeriesChunk> kept_chunks;
    kept_chunks.reserve(state.chunks.size());
    bool touched = false;
    for (auto &chunk : state.chunks) {
      const size_t points_before = chunk.points.size();
      chunk.points.erase(std::remove_if(chunk.points.begin(), chunk.points.end(), [&](const SeriesPoint &point) {
        return point.t < cutoff_time;
      }), chunk.points.end());
      const size_t removed_points = points_before - chunk.points.size();
      result.series_points_removed += removed_points;
      touched = touched || removed_points > 0;

      if (chunk.points.empty()) {
        ++result.series_chunks_removed;
        touched = true;
        continue;
      }
      if (chunk.range.valid() && chunk.range.end >= cutoff_time) {
        if (chunk.range.start_ < cutoff_time) chunk.range.start_ = cutoff_time;
      } else {
        chunk.range = pointRange(chunk.points);
      }
      kept_chunks.push_back(std::move(chunk));
    }
    if (touched) result.touched_series_paths.push_back(path);
    state.chunks = std::move(kept_chunks);
    if (state.chunks.empty()) empty_series_paths.push_back(path);
  }

  for (const std::string &path : empty_series_paths) {
    series_.erase(path);
    ++result.series_paths_removed;
  }

  std::vector<MessageId> empty_can_ids;
  for (auto &[id, state] : can_events_) {
    const size_t events_before = state.events.size();
    auto first_kept = std::lower_bound(state.events.begin(), state.events.end(), cutoff_time,
                                       [](const CanEvent &event, double t) { return event.mono_time < t; });
    result.can_events_removed += static_cast<size_t>(std::distance(state.events.begin(), first_kept));
    state.events.erase(state.events.begin(), first_kept);
    auto [coverage, removed_coverage_ranges] = trimRangesBefore(std::move(state.coverage), cutoff_time);
    state.coverage = std::move(coverage);
    result.coverage_ranges_removed += removed_coverage_ranges;
    if (state.events.empty()) {
      empty_can_ids.push_back(id);
    } else if (events_before != state.events.size()) {
      state.coverage = normalize_ranges(std::move(state.coverage));
    }
  }
  for (const MessageId &id : empty_can_ids) {
    can_events_.erase(id);
    ++result.can_messages_removed;
  }

  auto [coverage, removed_coverage_ranges] = trimRangesBefore(std::move(coverage_), cutoff_time);
  coverage_ = std::move(coverage);
  result.coverage_ranges_removed += removed_coverage_ranges;

  std::sort(result.touched_series_paths.begin(), result.touched_series_paths.end());
  result.touched_series_paths.erase(std::unique(result.touched_series_paths.begin(), result.touched_series_paths.end()),
                                    result.touched_series_paths.end());
  if (result.series_paths_removed > 0 || result.series_chunks_removed > 0 || result.series_points_removed > 0 ||
      result.can_messages_removed > 0 || result.can_events_removed > 0 || result.coverage_ranges_removed > 0) {
    ++generation_;
  }
  return result;
}

void Store::clear() {
  {
    // Only staged batches are shared; clear ownership for the rest is synchronized by caller thread usage.
    std::lock_guard lock(staged_mutex_);
    staged_batches_.clear();
  }
  series_.clear();
  can_events_.clear();
  coverage_.clear();
  ++generation_;
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
      const bool zero_span = chunk.range.span() <= EPS;
      if (zero_span) {
        if (chunk.range.start_ < view.requested.start_ || chunk.range.start_ > view.requested.end) continue;
      } else if (!intersects(chunk.range, view.requested)) {
        continue;
      }
      ranges.push_back(chunk.range);
    }
    for (const auto &point : chunk.points) {
      if (point.t >= view.requested.start_ && point.t <= view.requested.end) points.push_back(point);
    }
  }

  view.total_points = points.size();
  view.decimated = points.size() > max_points;
  view.points = decimate(std::move(points), max_points);
  view.coverage = coverageFor(view.requested, std::move(ranges));
  return view;
}

SeriesView Store::series_full(std::string_view path, TimeRange range) const {
  return series(path, range.start_, range.end, std::numeric_limits<size_t>::max());
}

CanEventView Store::can_events(const MessageId &id, TimeRange range) const {
  CanEventView view;
  view.id = id;
  view.requested = orderedRange(range.start_, range.end);
  view.coverage.requested = view.requested;

  const auto it = can_events_.find(id);
  if (it == can_events_.end()) return view;

  const auto &events = it->second.events;
  auto first = std::lower_bound(events.begin(), events.end(), view.requested.start_,
                                [](const CanEvent &event, double t) { return event.mono_time < t; });
  for (auto e = first; e != events.end() && e->mono_time <= view.requested.end; ++e) {
    view.events.push_back(*e);
  }
  view.coverage = coverageFor(view.requested, it->second.coverage);
  return view;
}

CanSummaryView Store::can_event_summary(const MessageId &id, TimeRange range, bool with_data) const {
  CanSummaryView view;
  view.id = id;
  view.requested = orderedRange(range.start_, range.end);
  view.coverage.requested = view.requested;

  const auto it = can_events_.find(id);
  if (it == can_events_.end()) return view;

  const auto &events = it->second.events;
  auto first = std::lower_bound(events.begin(), events.end(), view.requested.start_,
                                [](const CanEvent &event, double t) { return event.mono_time < t; });
  auto last = std::upper_bound(first, events.end(), view.requested.end,
                               [](double t, const CanEvent &event) { return t < event.mono_time; });
  view.count = static_cast<size_t>(std::distance(first, last));
  if (first != last) {
    view.first_time = first->mono_time;
    const auto latest = std::prev(last);
    view.last_time = latest->mono_time;
    if (with_data) view.latest_data = latest->data;
  }
  view.coverage = coverageFor(view.requested, it->second.coverage);
  return view;
}

std::vector<double> Store::byte_change_times(const MessageId &id, TimeRange range, size_t byte_count) const {
  std::vector<double> last_change(byte_count, -std::numeric_limits<double>::infinity());
  const auto it = can_events_.find(id);
  if (it == can_events_.end() || byte_count == 0) return last_change;

  const TimeRange wanted = orderedRange(range.start_, range.end);
  const auto &events = it->second.events;
  auto first = std::lower_bound(events.begin(), events.end(), wanted.start_,
                                [](const CanEvent &event, double t) { return event.mono_time < t; });
  auto last = std::upper_bound(first, events.end(), wanted.end,
                               [](double t, const CanEvent &event) { return t < event.mono_time; });
  size_t resolved = 0;
  for (auto e = last; e != first && resolved < byte_count;) {
    --e;
    if (e == first) break;
    const std::vector<uint8_t> &cur = e->data;
    const std::vector<uint8_t> &prev = std::prev(e)->data;
    const size_t n = std::min({cur.size(), prev.size(), byte_count});
    for (size_t b = 0; b < n; ++b) {
      if (!std::isfinite(last_change[b]) && cur[b] != prev[b]) {
        last_change[b] = e->mono_time;
        ++resolved;
      }
    }
  }
  return last_change;
}

size_t Store::staged_batch_count() const {
  // Sample staged count under mutex because stage() can append concurrently.
  std::lock_guard lock(staged_mutex_);
  return staged_batches_.size();
}

std::vector<std::string> Store::series_paths() const {
  std::vector<std::string> paths;
  paths.reserve(series_.size());
  for (const auto &[path, _] : series_) {
    paths.push_back(path);
  }
  std::sort(paths.begin(), paths.end());
  return paths;
}

std::vector<std::string> Store::series_paths_matching(std::string_view filter, size_t limit) const {
  std::vector<std::string> paths;
  if (limit == 0) return paths;
  paths.reserve(std::min(limit, series_.size()));

  auto append_path = [&](const std::string &path) {
    if (!filter.empty() && path.find(filter) == std::string::npos) return;
    if (paths.size() < limit) {
      paths.push_back(path);
      return;
    }
    auto largest = std::max_element(paths.begin(), paths.end());
    if (largest != paths.end() && path < *largest) *largest = path;
  };

  for (const auto &[path, _] : series_) {
    append_path(path);
  }
  std::sort(paths.begin(), paths.end());
  return paths;
}

std::vector<MessageId> Store::can_message_ids() const {
  std::vector<MessageId> ids;
  ids.reserve(can_events_.size());
  for (const auto &[id, _] : can_events_) {
    ids.push_back(id);
  }
  std::sort(ids.begin(), ids.end());
  return ids;
}

}  // namespace loggy
