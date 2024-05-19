#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>
#include <utility>

#include <QThread>

#include "tools/replay/camera.h"
#include "tools/replay/route.h"

const QString DEMO_ROUTE = "a2a0ccea32023010|2023-07-27--13-01-19";

// one segment uses about 100M of memory
constexpr int MIN_SEGMENTS_CACHE = 5;

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

enum class FindFlag {
  nextEngagement,
  nextDisEngagement,
  nextUserFlag,
  nextInfo,
  nextWarning,
  nextCritical
};

enum class TimelineType { None, Engaged, AlertInfo, AlertWarning, AlertCritical, UserFlag };
typedef bool (*replayEventFilter)(const Event *, void *);
Q_DECLARE_METATYPE(std::shared_ptr<LogReader>);

class Replay : public QObject {
  Q_OBJECT

public:
  Replay(QString route, QStringList allow, QStringList block, SubMaster *sm = nullptr,
         uint32_t flags = REPLAY_FLAG_NONE, QString data_dir = "", QObject *parent = 0);
  ~Replay();
  bool load();
  void start(int seconds = 0);
  void pause(bool pause);
  void seekToFlag(FindFlag flag);
  void seekTo(double seconds, bool relative);
  inline bool isPaused() const { return user_paused_; }
  // the filter is called in streaming thread.try to return quickly from it to avoid blocking streaming.
  // the filter function must return true if the event should be filtered.
  // otherwise it must return false.
  inline void installEventFilter(replayEventFilter filter, void *opaque) {
    filter_opaque = opaque;
    event_filter = filter;
  }
  inline int segmentCacheLimit() const { return segment_cache_limit; }
  inline void setSegmentCacheLimit(int n) { segment_cache_limit = std::max(MIN_SEGMENTS_CACHE, n); }
  inline bool hasFlag(REPLAY_FLAGS flag) const { return flags_ & flag; }
  inline void addFlag(REPLAY_FLAGS flag) { flags_ |= flag; }
  inline void removeFlag(REPLAY_FLAGS flag) { flags_ &= ~flag; }
  inline const Route* route() const { return route_.get(); }
  inline double currentSeconds() const { return double(cur_mono_time_ - route_start_ts_) / 1e9; }
  inline QDateTime routeDateTime() const { return route_date_time_; }
  inline QDateTime currentDateTime() const { return route_date_time_.addSecs(currentSeconds()); }
  inline uint64_t routeStartTime() const { return route_start_ts_; }
  inline double toSeconds(uint64_t mono_time) const { return (mono_time - route_start_ts_) / 1e9; }
  inline int totalSeconds() const { return (!segments_.empty()) ? (segments_.rbegin()->first + 1) * 60 : 0; }
  inline void setSpeed(float speed) { speed_ = speed; }
  inline float getSpeed() const { return speed_; }
  inline const std::vector<Event> *events() const { return &events_; }
  inline const std::map<int, std::unique_ptr<Segment>> &segments() const { return segments_; }
  inline const std::string &carFingerprint() const { return car_fingerprint_; }
  inline const std::vector<std::tuple<double, double, TimelineType>> getTimeline() {
    std::lock_guard lk(timeline_lock);
    return timeline;
  }

signals:
  void streamStarted();
  void segmentsMerged();
  void seekedTo(double sec);
  void qLogLoaded(int segnum, std::shared_ptr<LogReader> qlog);

protected slots:
  void segmentLoadFinished(bool success);

protected:
  typedef std::map<int, std::unique_ptr<Segment>> SegmentMap;
  std::optional<uint64_t> find(FindFlag flag);
  void pauseStreamThread();
  void startStream(const Segment *cur_segment);
  void streamThread();
  void updateSegmentsCache();
  void loadSegmentInRange(SegmentMap::iterator begin, SegmentMap::iterator cur, SegmentMap::iterator end);
  void mergeSegments(const SegmentMap::iterator &begin, const SegmentMap::iterator &end);
  void updateEvents(const std::function<bool()>& update_events_function);
  std::vector<Event>::const_iterator publishEvents(std::vector<Event>::const_iterator first,
                                                   std::vector<Event>::const_iterator last);
  void publishMessage(const Event *e);
  void publishFrame(const Event *e);
  void buildTimeline();
  inline bool isSegmentMerged(int n) const { return merged_segments_.count(n) > 0; }

  pthread_t stream_thread_id = 0;
  QThread *stream_thread_ = nullptr;
  std::mutex stream_lock_;
  bool user_paused_ = false;
  std::condition_variable stream_cv_;
  std::atomic<int> current_segment_ = 0;
  double seeking_to_seconds_ = -1;
  SegmentMap segments_;
  // the following variables must be protected with stream_lock_
  std::atomic<bool> exit_ = false;
  std::atomic<bool> paused_ = false;
  bool events_ready_ = false;
  QDateTime route_date_time_;
  uint64_t route_start_ts_ = 0;
  std::atomic<uint64_t> cur_mono_time_ = 0;
  std::vector<Event> events_;
  std::set<int> merged_segments_;

  // messaging
  SubMaster *sm = nullptr;
  std::unique_ptr<PubMaster> pm;
  std::vector<const char*> sockets_;
  std::vector<bool> filters_;
  std::unique_ptr<Route> route_;
  std::unique_ptr<CameraServer> camera_server_;
  std::atomic<uint32_t> flags_ = REPLAY_FLAG_NONE;

  std::mutex timeline_lock;
  QFuture<void> timeline_future;
  std::vector<std::tuple<double, double, TimelineType>> timeline;
  std::string car_fingerprint_;
  std::atomic<float> speed_ = 1.0;
  replayEventFilter event_filter = nullptr;
  void *filter_opaque = nullptr;
  int segment_cache_limit = MIN_SEGMENTS_CACHE;
};
