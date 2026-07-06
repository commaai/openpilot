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
  std::string road_camera_path;
  std::string driver_camera_path;
  std::string wide_road_camera_path;
  std::string qroad_camera_path;
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

  void set_store(Store *store);
  void set_route_segments(std::vector<RouteSegment> segments);
  std::vector<RouteSegment> segments() const;

  void set_tracker_time(double seconds);
  double tracker_time() const;
  void set_visible_ranges(std::vector<TimeRange> ranges);
  std::vector<TimeRange> visible_ranges() const;

  std::vector<SegmentPriority> priority_order() const;
  std::optional<SegmentWorkItem> take_next();

  void mark_pending(int segment);
  void mark_in_flight(int segment);
  void mark_loaded(int segment);
  void mark_failed(int segment, std::string error);

  bool publish(StoreBatch batch);

private:
  RouteSegment *find_segment_locked(int segment);
  const RouteSegment *find_segment_locked(int segment) const;

  // Protects route segment state plus scheduler priorities; Store is only borrowed long enough to publish.
  mutable std::mutex mutex_;
  Store *store_ = nullptr;
  std::vector<RouteSegment> segments_;
  std::vector<TimeRange> visible_ranges_;
  double tracker_time_ = 0.0;
};

}  // namespace loggy
