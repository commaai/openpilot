#pragma once

#include <algorithm>
#include <atomic>
#include <memory>
#include <deque>
#include <thread>
#include <vector>

#include "tools/cabana/streams/abstractstream.h"

class LiveStream : public AbstractStream {
public:
  LiveStream();
  virtual ~LiveStream();
  void start() override;
  bool supportsCereal() const override { return true; }
  std::vector<std::string> cerealSignalNames() const override;
  cabana::CerealReader cerealReader() const override;
  cabana::CerealHistoryReader cerealHistoryReader(bool entire_route = false) const override;
  cabana::RecordedCameraReader cameraReader(const std::string &view) const override;
  void stop();
  inline std::chrono::system_clock::time_point beginDateTime() const override { return begin_date_time; }
  inline uint64_t beginMonoTime() const override { return begin_event_ts; }
  double maxSeconds() const override { return std::max(1.0, (lastest_event_ts - begin_event_ts) / 1e9); }
  void setSpeed(float speed) override { speed_ = speed; }
  double getSpeed() override { return speed_; }
  bool isPaused() const override { return paused_; }
  void pause(bool pause) override;
  void seekTo(double sec) override;

protected:
  virtual void streamThread() = 0;
  void handleEvent(kj::ArrayPtr<capnp::word> event);

  std::atomic<bool> exit_ = false;

private:
  void updateThread();
  void updateLastMessages() override;
  void updateEvents();

  struct HistorySpool;
  std::shared_ptr<HistorySpool> history_;
  struct CerealEvent {
    uint64_t time;
    std::string service;
    std::vector<capnp::word> words;
  };
  mutable std::mutex cereal_mutex_;
  std::deque<std::shared_ptr<const CerealEvent>> cereal_events_;
  cabana::CerealDiscovery cereal_discovery_;
  size_t cereal_bytes_ = 0;
  std::deque<cabana::RecordedFrame> thumbnails_;
  uint64_t first_received_ts_ = 0;
  uint64_t latest_received_ts_ = 0;
  bool cereal_dirty_ = false;
  std::mutex lock;
  std::thread stream_thread, update_thread;
  std::atomic<bool> update_pending_ = false;
  std::vector<const CanEvent *> received_events_;

  std::chrono::system_clock::time_point begin_date_time;
  uint64_t begin_event_ts = 0;
  uint64_t lastest_event_ts = 0;
  uint64_t current_event_ts = 0;
  uint64_t first_event_ts = 0;
  uint64_t first_update_ts = 0;
  bool post_last_event = true;
  double speed_ = 1;
  double previous_speed_ = 1;
  bool paused_ = false;

  struct Logger;
  std::unique_ptr<Logger> logger;
};
