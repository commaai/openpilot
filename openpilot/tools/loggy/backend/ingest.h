#pragma once

#include <optional>
#include <mutex>
#include <string>
#include <vector>

#include "tools/loggy/backend/store.h"

namespace loggy {

enum class SegmentState {
  Pending,
  InFlight,
  Loaded,
  Failed,
};

struct RouteSegment {
  int segment = -1;
  TimeRange range;
  std::string log_path;
  std::string cache_path;
  SegmentState state = SegmentState::Pending;
  std::string error;
};

struct SegmentPriority {
  int segment = -1;
  TimeRange range;
  SegmentState state = SegmentState::Pending;
  bool intersects_visible = false;
  bool contains_tracker = false;
  double visible_distance = 0.0;
  double tracker_distance = 0.0;
};

struct SegmentWorkItem {
  int segment = -1;
  TimeRange range;
  std::string log_path;
  std::string cache_path;
};

class SegmentScheduler {
public:
  explicit SegmentScheduler(Store *store = nullptr);

  void setStore(Store *store);
  void setRouteSegments(std::vector<RouteSegment> segments);
  std::vector<RouteSegment> segments() const;

  void setTrackerTime(double seconds);
  double trackerTime() const;
  void setVisibleRanges(std::vector<TimeRange> ranges);
  std::vector<TimeRange> visibleRanges() const;

  std::vector<SegmentPriority> priorityOrder() const;
  std::optional<SegmentWorkItem> takeNext();

  void markPending(int segment);
  void markInFlight(int segment);
  void markLoaded(int segment);
  void markFailed(int segment, std::string error);

  bool publish(StoreBatch batch);

private:
  RouteSegment *findSegmentLocked(int segment);
  const RouteSegment *findSegmentLocked(int segment) const;

  mutable std::mutex mutex_;
  Store *store_ = nullptr;
  std::vector<RouteSegment> segments_;
  std::vector<TimeRange> visible_ranges_;
  double tracker_time_ = 0.0;
};

}  // namespace loggy
