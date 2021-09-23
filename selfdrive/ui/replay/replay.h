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
  Replay(QString route, QStringList allow, QStringList block, SubMaster *sm = nullptr, QObject *parent = 0);
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
  void updateEvents(const std::function<void()>& lambda);

  uint64_t route_start_ts = 0;
  uint64_t cur_mono_time_ = 0;
  std::atomic<int> current_segment = -1;

  QThread *thread;

  // logs
  std::mutex lock;
  std::atomic<bool> updating_events = false;
  std::atomic<bool> paused_ = false;
  bool events_updated = false;
  std::condition_variable stream_cv_;
  std::vector<Event *> *events = nullptr;
  std::unordered_map<uint32_t, EncodeIdx> *eidx = nullptr;
  std::vector<std::unique_ptr<Segment>> segments;
  std::vector<int> segments_merged;

  // messaging
  SubMaster *sm;
  PubMaster *pm;
  std::set<std::string> socks;
  VisionIpcServer *vipc_server = nullptr;
  std::unique_ptr<Route> route_;
  std::atomic<bool> exit_ = false;
};
