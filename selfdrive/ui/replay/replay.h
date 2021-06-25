#pragma once

#include <set>

#include "selfdrive/common/queue.h"
#include "selfdrive/common/util.h"
#include "selfdrive/ui/replay/camera.h"
#include "selfdrive/ui/replay/filereader.h"
#include "selfdrive/ui/replay/framereader.h"
#include "selfdrive/ui/replay/route.h"

class Segment : public QObject {
  Q_OBJECT

public:
  Segment(QObject *parent = nullptr) : QObject(parent) {}
  ~Segment();
  void load(int seg_num, const SegmentFile &file);
  inline bool loaded() const { return loading_ == 0 && failed_ == 0; }
  inline bool failed() const { return failed_ != 0; }
  int seg_num = 0;
  LogReader *log = nullptr;
  FrameReader *frames[MAX_CAMERAS] = {};

signals:
  void loadFinished(bool success);

private:
  std::atomic<int> loading_ = 0, failed_ = 0;
  std::thread frame_thread_[MAX_CAMERAS];
};

class Replay : public QObject {
  Q_OBJECT

public:
  Replay(SubMaster *sm = nullptr, QObject *parent = nullptr);
  ~Replay();
  bool start(const QString &routeName);
  bool start(const Route &route);
  void relativeSeek(int seconds);
  void seek(int seconds);
  void stop();
  inline bool running() { return stream_thread_.joinable(); }

public slots:
  void mergeEvents(bool success);

signals:
  void segmentChanged(int seg_num);

private:
  inline QString elapsedTime(uint64_t ns) {
    return QTime(0, 0, 0).addSecs((ns - route_start_ts_) / 1e9).toString("hh:mm:ss");
  }
  void seekTo(uint64_t to_ts);
  std::pair<int, int> cacheSegmentRange(int seg_num);
  void queueSegment(int seg_num);
  void streamThread();
  void pushFrame(int cur_seg_num, CameraType type, uint32_t frame_id);
  const Segment *getSegment(int segment);
  const std::string &eventSocketName(const Event *e);

  // messaging
  SubMaster *sm_ = nullptr;
  PubMaster *pm_ = nullptr;
  std::set<std::string> socks_;
  std::unordered_map<cereal::Event::Which, std::string> eventNameMap;

  // segments
  Route route_;
  std::mutex events_mutex_, segment_mutex_;
  std::condition_variable cv_;
  std::map<int, Segment*> segments_;
  std::unique_ptr<std::vector<Event *>> events_;

  uint64_t seek_ts_ = 0;
  std::atomic<int> current_segment_ = 0;
  std::atomic<uint64_t> route_start_ts_ = 0, current_ts_ = 0;  // ns

  // camera server
  CameraServer camera_server_;

  std::atomic<bool> exit_ = false, loading_events_ = false;
  std::thread stream_thread_;
};
