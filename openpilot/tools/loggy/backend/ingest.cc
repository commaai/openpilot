#include "tools/loggy/backend/ingest.h"

#include <algorithm>
#include <limits>
#include <mutex>
#include <utility>

namespace loggy {
namespace {

double visibleDistance(TimeRange segment, const std::vector<TimeRange> &visible_ranges) {
  if (visible_ranges.empty()) return std::numeric_limits<double>::infinity();
  double best = std::numeric_limits<double>::infinity();
  for (const auto &range : visible_ranges) {
    best = std::min(best, distance(segment, range));
  }
  return best;
}

SegmentPriority makePriority(const RouteSegment &segment, double tracker_time, const std::vector<TimeRange> &visible_ranges) {
  SegmentPriority priority;
  priority.segment = segment.segment;
  priority.range = segment.range;
  priority.state = segment.state;
  priority.visible_distance = visibleDistance(segment.range, visible_ranges);
  priority.tracker_distance = distance_to_point(segment.range, tracker_time);
  priority.intersects_visible = priority.visible_distance == 0.0;
  priority.contains_tracker = priority.tracker_distance == 0.0;
  return priority;
}

bool priorityLess(const SegmentPriority &a, const SegmentPriority &b, bool has_visible_ranges) {
  if (has_visible_ranges && a.intersects_visible != b.intersects_visible) {
    return a.intersects_visible;
  }
  if (a.contains_tracker != b.contains_tracker) {
    return a.contains_tracker;
  }

  const double a_distance = has_visible_ranges ? std::min(a.visible_distance, a.tracker_distance) : a.tracker_distance;
  const double b_distance = has_visible_ranges ? std::min(b.visible_distance, b.tracker_distance) : b.tracker_distance;
  if (a_distance != b_distance) return a_distance < b_distance;
  if (a.tracker_distance != b.tracker_distance) return a.tracker_distance < b.tracker_distance;
  return a.segment < b.segment;
}

SegmentWorkItem workItem(const RouteSegment &segment) {
  return {
    .segment = segment.segment,
    .range = segment.range,
    .log_path = segment.log_path,
    .cache_path = segment.cache_path,
  };
}

}  // namespace

SegmentScheduler::SegmentScheduler(Store *store) : store_(store) {}

void SegmentScheduler::set_store(Store *store) {
  // Guards scheduler+store pointer handoff so take_next()/publish see a consistent target.
  std::lock_guard lock(mutex_);
  store_ = store;
}

void SegmentScheduler::set_route_segments(std::vector<RouteSegment> segments) {
  std::sort(segments.begin(), segments.end(), [](const auto &a, const auto &b) {
    return a.segment < b.segment;
  });
  // Sort first, then lock briefly for swap to keep publication of a new segment set atomic.
  std::lock_guard lock(mutex_);
  segments_ = std::move(segments);
}

std::vector<RouteSegment> SegmentScheduler::segments() const {
  // Return copy so callers can iterate without holding the scheduler mutex.
  std::lock_guard lock(mutex_);
  return segments_;
}

void SegmentScheduler::set_tracker_time(double seconds) {
  // tracker_time_ is only observed during scheduling; keep update synchronized with priority reads.
  std::lock_guard lock(mutex_);
  tracker_time_ = seconds;
}

double SegmentScheduler::tracker_time() const {
  // Read copy under lock to prevent torn reads while scheduler threads update state.
  std::lock_guard lock(mutex_);
  return tracker_time_;
}

void SegmentScheduler::set_visible_ranges(std::vector<TimeRange> ranges) {
  // Normalize+store under lock so priority computation sees a complete range set.
  std::lock_guard lock(mutex_);
  visible_ranges_ = normalize_ranges(std::move(ranges));
}

std::vector<TimeRange> SegmentScheduler::visible_ranges() const {
  // Copy visible ranges for caller so this mutex can stay short-lived.
  std::lock_guard lock(mutex_);
  return visible_ranges_;
}

std::vector<SegmentPriority> SegmentScheduler::priority_order() const {
  std::lock_guard lock(mutex_);
  std::vector<SegmentPriority> order;
  order.reserve(segments_.size());
  for (const auto &segment : segments_) {
    if (segment.state != SegmentState::Pending) continue;
    order.push_back(makePriority(segment, tracker_time_, visible_ranges_));
  }

  const bool has_visible_ranges = !visible_ranges_.empty();
  std::sort(order.begin(), order.end(), [has_visible_ranges](const auto &a, const auto &b) {
    return priorityLess(a, b, has_visible_ranges);
  });
  return order;
}

std::optional<SegmentWorkItem> SegmentScheduler::take_next() {
  // Entire selection-and-state-change is one critical section to prevent duplicate segment assignment.
  std::lock_guard lock(mutex_);
  std::vector<SegmentPriority> order;
  order.reserve(segments_.size());
  for (const auto &segment : segments_) {
    if (segment.state != SegmentState::Pending) continue;
    order.push_back(makePriority(segment, tracker_time_, visible_ranges_));
  }
  const bool has_visible_ranges = !visible_ranges_.empty();
  std::sort(order.begin(), order.end(), [has_visible_ranges](const auto &a, const auto &b) {
    return priorityLess(a, b, has_visible_ranges);
  });
  if (order.empty()) return std::nullopt;

  RouteSegment *segment = find_segment_locked(order.front().segment);
  if (segment == nullptr) return std::nullopt;
  segment->state = SegmentState::InFlight;
  segment->error.clear();
  return workItem(*segment);
}

void SegmentScheduler::mark_pending(int segment) {
  std::lock_guard lock(mutex_);
  if (auto *record = find_segment_locked(segment)) {
    record->state = SegmentState::Pending;
    record->error.clear();
  }
}

void SegmentScheduler::mark_in_flight(int segment) {
  std::lock_guard lock(mutex_);
  if (auto *record = find_segment_locked(segment)) {
    record->state = SegmentState::InFlight;
    record->error.clear();
  }
}

void SegmentScheduler::mark_loaded(int segment) {
  std::lock_guard lock(mutex_);
  if (auto *record = find_segment_locked(segment)) {
    record->state = SegmentState::Loaded;
    record->error.clear();
  }
}

void SegmentScheduler::mark_failed(int segment, std::string error) {
  std::lock_guard lock(mutex_);
  if (auto *record = find_segment_locked(segment)) {
    record->state = SegmentState::Failed;
    record->error = std::move(error);
  }
}

bool SegmentScheduler::publish(StoreBatch batch) {
  // Resolve segment state while holding mutex, then publish outside it to avoid re-entrant lock nesting with store.
  Store *store = nullptr;
  {
    std::lock_guard lock(mutex_);
    RouteSegment *segment = find_segment_locked(batch.segment);
    if (segment != nullptr) {
      if (batch.coverage.empty() && segment->range.valid()) batch.coverage.push_back(segment->range);
      segment->state = SegmentState::Loaded;
      segment->error.clear();
    }
    store = store_;
  }

  if (store == nullptr) return false;
  store->stage(std::move(batch));
  return true;
}

RouteSegment *SegmentScheduler::find_segment_locked(int segment) {
  // Called with mutex_ held by caller.
  auto it = std::find_if(segments_.begin(), segments_.end(), [segment](const auto &record) {
    return record.segment == segment;
  });
  return it == segments_.end() ? nullptr : &*it;
}

const RouteSegment *SegmentScheduler::find_segment_locked(int segment) const {
  // Called with mutex_ held by caller.
  auto it = std::find_if(segments_.begin(), segments_.end(), [segment](const auto &record) {
    return record.segment == segment;
  });
  return it == segments_.end() ? nullptr : &*it;
}

}  // namespace loggy
