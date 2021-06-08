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
  Segment(int seg_num, const SegmentFile &file, QObject *parent = nullptr);

  const int seg_num;
  LogReader *log = nullptr;
  std::shared_ptr<FrameReader> frames[MAX_CAMERAS] = {};
  std::atomic<bool> loaded = false;

signals:
  void finishedRead();

private:
  std::atomic<int> loading_ = 0;
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
  bool running() { return stream_thread_.joinable(); }

public slots:
  void mergeEvents();

private:
  QString elapsedTime(uint64_t ns);
  void seekTo(uint64_t to_ts);
  std::pair<int, int> queueSegmentRange();
  void queueSegmentThread();
  void streamThread();
  void pushFrame(int cur_seg_num, CameraType type, uint32_t frame_id);
  const Segment* getSegment(int segment);
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
  std::map<int, std::unique_ptr<Segment>> segments_;
  std::unique_ptr<std::vector<Event *>> events_;

  uint64_t seek_ts_ = 0;
  std::atomic<int> current_segment_ = 0;
  std::atomic<uint64_t> route_start_ts_, current_ts_ = 0;  // ns

  // camera server
  CameraServer camera_server_;

  std::atomic<bool> exit_ = false, loading_events_ = false;
  std::thread stream_thread_, queue_thread_;
};
