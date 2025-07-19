#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "tools/replay/camera.h"
#include "tools/replay/seg_mgr.h"
#include "tools/replay/timeline.h"

#define DEMO_ROUTE "a2a0ccea32023010|2023-07-27--13-01-19"

enum REPLAY_FLAGS {
  REPLAY_FLAG_NONE = 0x0000,
  REPLAY_FLAG_DCAM = 0x0002,
  REPLAY_FLAG_ECAM = 0x0004,
  REPLAY_FLAG_NO_LOOP = 0x0010,
  REPLAY_FLAG_NO_FILE_CACHE = 0x0020,
  REPLAY_FLAG_QCAMERA = 0x0040,
  REPLAY_FLAG_NO_HW_DECODER = 0x0100,
  REPLAY_FLAG_NO_VIPC = 0x0400,
  REPLAY_FLAG_ALL_SERVICES = 0x0800,
};

class Replay {
public:
  Replay(const std::string &route, std::vector<std::string> allow, std::vector<std::string> block, SubMaster *sm = nullptr,
         uint32_t flags = REPLAY_FLAG_NONE, const std::string &data_dir = "", bool auto_source = false);
  ~Replay();
  bool load();
  RouteLoadError lastRouteError() const { return route().lastError(); }
  void start(int seconds = 0) { seekTo(min_seconds_ + seconds, false); }
  void pause(bool pause);
  void seekToFlag(FindFlag flag);
  void seekTo(double seconds, bool relative);
  inline bool isPaused() const { return user_paused_; }
  inline int segmentCacheLimit() const { return seg_mgr_->segment_cache_limit_; }
  inline void setSegmentCacheLimit(int n) { seg_mgr_->segment_cache_limit_ = std::max(MIN_SEGMENTS_CACHE, n); }
  inline bool hasFlag(REPLAY_FLAGS flag) const { return flags_ & flag; }
  void setLoop(bool loop) { loop ? flags_ &= ~REPLAY_FLAG_NO_LOOP : flags_ |= REPLAY_FLAG_NO_LOOP; }
  bool loop() const { return !(flags_ & REPLAY_FLAG_NO_LOOP); }
  const Route &route() const { return seg_mgr_->route_; }
  inline double currentSeconds() const { return double(cur_mono_time_ - route_start_ts_) / 1e9; }
  inline std::time_t routeDateTime() const { return route_date_time_; }
  inline uint64_t routeStartNanos() const { return route_start_ts_; }
  inline double toSeconds(uint64_t mono_time) const { return (mono_time - route_start_ts_) / 1e9; }
  inline double minSeconds() const { return min_seconds_; }
  inline double maxSeconds() const { return max_seconds_; }
  inline void setSpeed(float speed) { speed_ = speed; }
  inline float getSpeed() const { return speed_; }
  inline const std::string &carFingerprint() const { return car_fingerprint_; }
  inline const std::shared_ptr<std::vector<Timeline::Entry>> getTimeline() const { return timeline_.getEntries(); }
  inline const std::optional<Timeline::Entry> findAlertAtTime(double sec) const { return timeline_.findAlertAtTime(sec); }
  const std::shared_ptr<SegmentManager::EventData> getEventData() const { return seg_mgr_->getEventData(); }
  void installEventFilter(std::function<bool(const Event *)> filter) { event_filter_ = filter; }

  // Event callback functions
  std::function<void()> onSegmentsMerged = nullptr;
  std::function<void(double)> onSeeking = nullptr;
  std::function<void(double)> onSeekedTo = nullptr;
  std::function<void(std::shared_ptr<LogReader>)> onQLogLoaded = nullptr;

private:
  void setupServices(const std::vector<std::string> &allow, const std::vector<std::string> &block);
  void setupSegmentManager(bool has_filters);
  void startStream(const std::shared_ptr<Segment> segment);
  void streamThread();
  void handleSegmentMerge();
  void interruptStream(const std::function<bool()>& update_fn);
  std::vector<Event>::const_iterator publishEvents(std::vector<Event>::const_iterator first,
                                                   std::vector<Event>::const_iterator last);
  void publishMessage(const Event *e);
  void publishFrame(const Event *e);
  void checkSeekProgress();

  std::unique_ptr<SegmentManager> seg_mgr_;
  Timeline timeline_;

  pthread_t stream_thread_id = 0;
  std::thread stream_thread_;
  std::mutex stream_lock_;
  bool user_paused_ = false;
  std::condition_variable stream_cv_;
  std::atomic<int> current_segment_ = 0;
  std::atomic<double> seeking_to_ = -1.0;
  std::atomic<bool> exit_ = false;
  std::atomic<bool> interrupt_requested_ = false;
  bool events_ready_ = false;
  std::time_t route_date_time_;
  uint64_t route_start_ts_ = 0;
  std::atomic<uint64_t> cur_mono_time_ = 0;
  cereal::Event::Which cur_which_ = cereal::Event::Which::INIT_DATA;
  double min_seconds_ = 0;
  double max_seconds_ = 0;
  SubMaster *sm_ = nullptr;
  std::unique_ptr<PubMaster> pm_;
  std::vector<const char*> sockets_;
  std::unique_ptr<CameraServer> camera_server_;
  std::atomic<uint32_t> flags_ = REPLAY_FLAG_NONE;

  std::string car_fingerprint_;
  std::atomic<float> speed_ = 1.0;
  std::function<bool(const Event *)> event_filter_ = nullptr;

  std::shared_ptr<SegmentManager::EventData> event_data_ = std::make_shared<SegmentManager::EventData>();
};
