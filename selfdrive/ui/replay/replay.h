#pragma once

#include <QThread>

#include <capnp/dynamic.h>
#include "cereal/visionipc/visionipc_server.h"
#include "selfdrive/ui/replay/camera.h"
#include "selfdrive/ui/replay/route.h"

constexpr int FORWARD_SEGS = 2;
constexpr int BACKWARD_SEGS = 2;

class Replay : public QObject {
  Q_OBJECT

public:
  enum Flags {
    None = 0,
    LoadDriverCam = 0x1,
    LoadWideRoadCam = 0x2,
    FallbackToQLog = 0x4
  };

  Replay(QString route, QStringList allow, QStringList block, SubMaster *sm = nullptr, uint32_t flag = None, QObject *parent = 0);
  ~Replay();
  bool load();
  void start(int seconds = 0);
  bool isPaused() const { return paused_; }
  void pause(bool pause);

signals:
 void segmentChanged();
 void seekTo(int seconds, bool relative);

protected slots:
  void queueSegment();
  void doSeek(int seconds, bool relative);

protected:
  void stream();
  void setCurrentSegment(int n);
  void mergeSegments(int begin_idx, int end_idx);
  void updateEvents(const std::function<bool()>& lambda);
  void publishFrame(const Event *e);
  inline bool isSegmentLoaded(int n) { return segments_[n] && segments_[n]->isLoaded(); }

  // logs
  std::mutex stream_lock_;
  std::condition_variable stream_cv_;
  std::atomic<bool> updating_events_ = false;
  std::atomic<int> current_segment_ = -1;
  std::unique_ptr<Route> route_;
  std::vector<std::unique_ptr<Segment>> segments_;
  std::unique_ptr<CameraServer> camera_server_;
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

  uint32_t flags_ = None;
  QThread *stream_thread_ = nullptr;
};
