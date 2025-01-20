#include "tools/replay/seg_mgr.h"

#include <algorithm>

SegmentManager::~SegmentManager() {
  {
    std::unique_lock lock(mutex_);
    exit_ = true;
    onSegmentMergedCallback_ = nullptr;
  }
  cv_.notify_one();
  if (thread_.joinable()) thread_.join();
}

bool SegmentManager::load() {
  if (!route_.load()) {
    rError("failed to load route: %s", route_.name().c_str());
    return false;
  }

  for (const auto &[n, file] : route_.segments()) {
    if (!file.rlog.empty() || !file.qlog.empty()) {
      segments_.insert({n, nullptr});
    }
  }

  if (segments_.empty()) {
    rInfo("no valid segments in route: %s", route_.name().c_str());
    return false;
  }

  rInfo("loaded route %s with %zu valid segments", route_.name().c_str(), segments_.size());
  thread_ = std::thread(&SegmentManager::manageSegmentCache, this);
  return true;
}

void SegmentManager::setCurrentSegment(int seg_num) {
  {
    std::unique_lock lock(mutex_);
    cur_seg_num_ = seg_num;
    needs_update_ = true;
  }
  cv_.notify_one();
}

void SegmentManager::manageSegmentCache() {
  while (true) {
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [this]() { return exit_ || needs_update_; });
    if (exit_) break;

    needs_update_ = false;
    auto cur = segments_.lower_bound(cur_seg_num_);
    if (cur == segments_.end()) continue;

    // Calculate the range of segments to load
    auto begin = std::prev(cur, std::min<int>(segment_cache_limit_ / 2, std::distance(segments_.begin(), cur)));
    auto end = std::next(begin, std::min<int>(segment_cache_limit_, std::distance(begin, segments_.end())));
    begin = std::prev(end, std::min<int>(segment_cache_limit_, std::distance(segments_.begin(), end)));

    loadSegmentsInRange(begin, cur, end);
    bool merged = mergeSegments(begin, end);

    // Free segments outside the current range
    std::for_each(segments_.begin(), begin, [](auto &segment) { segment.second.reset(); });
    std::for_each(end, segments_.end(), [](auto &segment) { segment.second.reset(); });

    lock.unlock();

    if (merged && onSegmentMergedCallback_) {
      onSegmentMergedCallback_();  // Notify listener that segments have been merged
    }
  }
}

bool SegmentManager::mergeSegments(const SegmentMap::iterator &begin, const SegmentMap::iterator &end) {
  std::set<int> segments_to_merge;
  size_t total_event_count = 0;
  for (auto it = begin; it != end; ++it) {
    const auto &segment = it->second;
    if (segment && segment->getState() == Segment::LoadState::Loaded) {
      segments_to_merge.insert(segment->seg_num);
      total_event_count += segment->log->events.size();
    }
  }

  if (segments_to_merge == merged_segments_) return false;

  auto merged_event_data = std::make_shared<EventData>();
  auto &merged_events = merged_event_data->events;
  merged_events.reserve(total_event_count);

  rDebug("merging segments: %s", join(segments_to_merge, ", ").c_str());
  for (int n : segments_to_merge) {
    const auto &events = segments_.at(n)->log->events;
    if (events.empty()) continue;

    // Skip INIT_DATA if present
    auto events_begin = (events.front().which == cereal::Event::Which::INIT_DATA) ? std::next(events.begin()) : events.begin();

    size_t previous_size = merged_events.size();
    merged_events.insert(merged_events.end(), events_begin, events.end());
    std::inplace_merge(merged_events.begin(), merged_events.begin() + previous_size, merged_events.end());

    merged_event_data->segments[n] = segments_.at(n);
  }

  std::atomic_store(&event_data_, std::move(merged_event_data));
  merged_segments_ = segments_to_merge;

  return true;
}

void SegmentManager::loadSegmentsInRange(SegmentMap::iterator begin, SegmentMap::iterator cur, SegmentMap::iterator end) {
  auto tryLoadSegment = [this](auto first, auto last) {
    for (auto it = first; it != last; ++it) {
      auto &segment_ptr = it->second;
      if (!segment_ptr) {
        segment_ptr = std::make_shared<Segment>(
            it->first, route_.at(it->first), flags_, filters_,
            [this](int seg_num, bool success) { setCurrentSegment(cur_seg_num_); });
      }

      if (segment_ptr->getState() == Segment::LoadState::Loading) {
        return true;  // Segment is still loading
      }
    }
    return false;  // No segments need loading
  };

  // Try forward loading, then reverse if necessary
  if (!tryLoadSegment(cur, end)) {
    tryLoadSegment(std::make_reverse_iterator(cur), std::make_reverse_iterator(begin));
  }
}
