#pragma once

#include <QThread>

#include "selfdrive/ui/replay/camera.h"
#include "selfdrive/ui/replay/route.h"

constexpr int FORWARD_SEGS = 2;
constexpr int BACKWARD_SEGS = 1;

class Replay : public QObject {
  Q_OBJECT

public:
  Replay(QString route, QStringList allow, QStringList block, SubMaster *sm = nullptr, bool dcam = false, bool ecam = false,
         QString data_dir="", QObject *parent = 0);
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
  void startStream(const Segment *cur_segment);
  void stream();
  void setCurrentSegment(int n);
  void mergeSegments(const SegmentMap::iterator &begin, const SegmentMap::iterator &end);
  void updateEvents(const std::function<bool()>& lambda);
  void publishMessage(const Event *e);
  void publishFrame(const Event *e);
  inline int currentSeconds() const { return (cur_mono_time_ - route_start_ts_) / 1e9; }
  inline bool isSegmentLoaded(int n) {
    return std::find(segments_merged_.begin(), segments_merged_.end(), n) != segments_merged_.end();
  }

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
  bool load_dcam = false, load_ecam = false;
  std::unique_ptr<CameraServer> camera_server_;
};
