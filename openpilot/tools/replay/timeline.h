#pragma once

#include <atomic>
#include <optional>
#include <thread>
#include <vector>

#include "tools/replay/route.h"

enum class TimelineType { None, Engaged, AlertInfo, AlertWarning, AlertCritical, UserBookmark };
enum class FindFlag { nextEngagement, nextDisEngagement, nextUserBookmark, nextInfo, nextWarning, nextCritical };

class Timeline {
public:
  struct Entry {
    double start_time;
    double end_time;
    TimelineType type;
    std::string text1;
    std::string text2;
  };

  Timeline() : timeline_entries_(std::make_shared<std::vector<Entry>>()) {}
  ~Timeline();

  void initialize(const Route &route, uint64_t route_start_ts, bool local_cache,
                  std::function<void(std::shared_ptr<LogReader>)> callback);
  std::optional<uint64_t> find(double cur_ts, FindFlag flag) const;
  std::optional<Entry> findAlertAtTime(double target_time) const;
  const std::shared_ptr<std::vector<Entry>> getEntries() const { return std::atomic_load(&timeline_entries_); }

private:
  void buildTimeline(const Route &route, uint64_t route_start_ts, bool local_cache,
                     std::function<void(std::shared_ptr<LogReader>)> callback);
  void updateEngagementStatus(const cereal::SelfdriveState::Reader &cs, std::optional<size_t> &idx, double seconds);
  void updateAlertStatus(const cereal::SelfdriveState::Reader &cs, std::optional<size_t> &idx, double seconds);

  std::thread thread_;
  std::atomic<bool> should_exit_ = false;

  // Temporarily holds entries before they are sorted and finalized
  std::vector<Entry> staging_entries_;

  // Final sorted timeline entries
  std::shared_ptr<std::vector<Entry>> timeline_entries_;
};
