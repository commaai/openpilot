#pragma once

#include <condition_variable>
#include <map>
#include <mutex>
#include <set>
#include <vector>

#include "tools/replay/route.h"

constexpr int MIN_SEGMENTS_CACHE = 5;

using SegmentMap = std::map<int, std::shared_ptr<Segment>>;

class SegmentManager {
public:
  struct EventData {
    std::vector<Event> events;  //  Events extracted from the segments
    SegmentMap segments;        // Associated segments that contributed to these events
    bool isSegmentLoaded(int n) const { return segments.find(n) != segments.end(); }
  };

  SegmentManager(const std::string &route_name, uint32_t flags, const std::string &data_dir = "")
      : flags_(flags), route_(route_name, data_dir), event_data_(std::make_shared<EventData>()) {}
  ~SegmentManager();

  bool load();
  void setCurrentSegment(int seg_num);
  void setCallback(const std::function<void()> &callback) { onSegmentMergedCallback_ = callback; }
  void setFilters(const std::vector<bool> &filters) { filters_ = filters; }
  const std::shared_ptr<EventData> getEventData() const { return std::atomic_load(&event_data_); }
  bool hasSegment(int n) const { return segments_.find(n) != segments_.end(); }

  Route route_;
  int segment_cache_limit_ = MIN_SEGMENTS_CACHE;

private:
  void manageSegmentCache();
  void loadSegmentsInRange(SegmentMap::iterator begin, SegmentMap::iterator cur, SegmentMap::iterator end);
  bool mergeSegments(const SegmentMap::iterator &begin, const SegmentMap::iterator &end);

  std::vector<bool> filters_;
  uint32_t flags_;

  std::mutex mutex_;
  std::condition_variable cv_;
  std::thread thread_;
  std::atomic<int> cur_seg_num_ = -1;
  bool needs_update_ = false;
  bool exit_ = false;

  SegmentMap segments_;
  std::shared_ptr<EventData> event_data_;
  std::function<void()> onSegmentMergedCallback_ = nullptr;
  std::set<int> merged_segments_;
};
