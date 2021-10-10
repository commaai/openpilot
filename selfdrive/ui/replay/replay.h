#pragma once

#include <QThread>

#include "selfdrive/ui/replay/camera.h"
#include "selfdrive/ui/replay/route.h"

constexpr int FORWARD_SEGS = 2;
constexpr int BACKWARD_SEGS = 1;

enum REPLAY_FLAGS {
  REPLAY_FLAG_NONE = 0x0000,
  REPLAY_FLAG_VERBOSE = 0x0001,
  REPLAY_FLAG_DCAM = 0x0002,
  REPLAY_FLAG_ECAM = 0x0004,
  REPLAY_FLAG_YUV = 0x0008,
  REPLAY_FLAG_NO_LOOP = 0x0010,
};

class Replay : public QObject {
  Q_OBJECT

public:
  Replay(QString route, QStringList allow, QStringList block, SubMaster *sm = nullptr, uint32_t flags = REPLAY_FLAG_NONE, QString data_dir = "", QObject *parent = 0);
  ~Replay();
  bool load();
  void start(int seconds = 0);
  void pause(bool pause);
  bool isPaused() const { return paused_; }

 signals:
  void segmentChanged();
  void seekTo(int seconds, bool relative);

 protected slots:
  void queueSegment();
  void doSeek(int seconds, bool relative);
  void segmentLoadFinished(bool sucess);

protected:
  typedef std::map<int, std::unique_ptr<Segment>> SegmentMap;
  void initRouteData(const std::vector<Event *> *events);
  void stream();
  void setCurrentSegment(int n);
  void mergeSegments(const SegmentMap::iterator &begin, const SegmentMap::iterator &end);
  void updateEvents(const std::function<bool()>& lambda);
  void publishMessage(const Event *e);
  void publishFrame(const Event *e);
  inline int currentSeconds() const { return (cur_mono_time_ - route_start_ts_) / 1e9; }

  QThread *stream_thread_ = nullptr;

  // logs
  std::mutex stream_lock_;
  std::condition_variable stream_cv_;
  std::atomic<bool> updating_events_ = false;
  std::atomic<int> current_segment_ = -1;
  SegmentMap segments_;
  // the following variables must be protected with stream_lock_
  bool exit_ = false;
  bool paused_ = false;
  bool events_updated_ = false;
  uint64_t route_start_ts_ = 0;
  uint64_t cur_mono_time_ = 0;
  std::vector<Event *> *events_ = nullptr;
  std::vector<int> segments_merged_;

  // messaging
  SubMaster *sm = nullptr;
  PubMaster *pm = nullptr;
  std::vector<const char*> sockets_;
  std::unique_ptr<Route> route_;
  std::unique_ptr<CameraServer> camera_server_;
  uint32_t flags_ = REPLAY_FLAG_NONE;
};
