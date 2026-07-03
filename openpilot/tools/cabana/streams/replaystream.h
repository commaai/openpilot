#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <set>

#include "common/prefix.h"
#include "tools/cabana/streams/abstractstream.h"
#include "tools/replay/replay.h"

class ReplayStream : public AbstractStream {
public:
  ReplayStream();
  ~ReplayStream();
  void start() override { replay->start(); }
  bool loadRoute(const std::string &route, const std::string &data_dir, uint32_t replay_flags = REPLAY_FLAG_NONE, bool auto_source = false);
  bool eventFilter(const Event *event);
  void seekTo(double ts) override { replay->seekTo(std::max(double(0), ts), false); }
  bool liveStreaming() const override { return false; }
  inline std::string routeName() const override { return replay->route().name(); }
  inline std::string carFingerprint() const override { return replay->carFingerprint(); }
  double minSeconds() const override { return replay->minSeconds(); }
  double maxSeconds() const override { return replay->maxSeconds(); }
  inline std::chrono::system_clock::time_point beginDateTime() const override { return std::chrono::system_clock::from_time_t(replay->routeDateTime()); }
  inline uint64_t beginMonoTime() const override { return replay->routeStartNanos(); }
  inline void setSpeed(float speed) override { replay->setSpeed(speed); }
  float getSpeed() const override { return replay->getSpeed(); }
  inline Replay *getReplay() const { return replay.get(); }
  inline bool isPaused() const override { return replay->isPaused(); }
  void pause(bool pause) override;

  cabana::Event<std::shared_ptr<LogReader>> qLogLoaded;

private:
  void mergeSegments();
  std::unique_ptr<Replay> replay = nullptr;
  std::set<int> processed_segments;
  std::unique_ptr<OpenpilotPrefix> op_prefix;

  // Blocks the replay thread inside onSegmentsMerged until the UI thread has
  // run mergeSegments() via the deferred-action queue (mirrors the old
  // Qt::BlockingQueuedConnection). aborting_ is set before the Replay is
  // destroyed so the wait can't deadlock the destructor's thread join.
  std::mutex merge_mutex_;
  std::condition_variable merge_cv_;
  bool merge_done_ = false;
  std::atomic<bool> aborting_ = false;
};
