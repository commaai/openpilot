#pragma once

#include <QThread>
#include <set>

#include <capnp/dynamic.h>
#include "cereal/visionipc/visionipc_server.h"
#include "selfdrive/ui/replay/route.h"

constexpr int FORWARD_SEGS = 2;
constexpr int BACKWARD_SEGS = 2;

class Replay : public QObject {
  Q_OBJECT

public:
  Replay(QString route, QStringList allow, QStringList block, SubMaster *sm = nullptr, bool dcam = false, bool ecam = false, QObject *parent = 0);
  ~Replay();

  void start(int seconds = 0);
  void seekTo(int seconds, bool relative = false);
  void relativeSeek(int seconds) { seekTo(seconds, true); }
  void pause(bool pause);
  bool isPaused() const { return paused_; }

signals:
 void segmentChanged(int);

protected slots:
  void queueSegment();

protected:
  void stream();
  void setCurrentSegment(int n);
  void mergeSegments(int begin_idx, int end_idx);
  void updateEvents(const std::function<bool()>& lambda);

  QThread *thread;

  // logs
  std::mutex lock_;
  std::condition_variable stream_cv_;
  std::atomic<bool> updating_events_ = false;
  std::atomic<int> current_segment_ = -1;
  bool exit_ = false;
  bool paused_ = false;
  bool events_updated_ = false;
  uint64_t route_start_ts_ = 0;
  uint64_t cur_mono_time_ = 0;
  std::vector<Event *> *events_ = nullptr;
  std::vector<std::unique_ptr<Segment>> segments_;
  std::vector<int> segments_merged_;

  // messaging
  SubMaster *sm;
  PubMaster *pm;
  std::set<std::string> socks;
  VisionIpcServer *vipc_server = nullptr;
  std::unique_ptr<Route> route_;
  bool load_dcam = false, load_ecam = false;
};
