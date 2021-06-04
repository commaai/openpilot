#pragma once

#include <set>

#include "cereal/visionipc/visionipc_server.h"
#include "selfdrive/common/queue.h"
#include "selfdrive/common/util.h"
#include "selfdrive/ui/replay/filereader.h"
#include "selfdrive/ui/replay/framereader.h"
#include "selfdrive/ui/replay/route.h"

class Segment : public QObject {
  Q_OBJECT

public:
  Segment(int seg_num, const SegmentFile &file, QObject *parent = nullptr);
  ~Segment();

  const int seg_num;
  LogReader *log = nullptr;
  FrameReader *frames[MAX_CAMERAS] = {};
  std::atomic<bool> loaded = false;
signals:
  void finishedRead();

private:
  std::atomic<int> loading_ = 0;
};

class CameraServer {
public:
  CameraServer();
  ~CameraServer();
  void stop();
  void ensureServerForSegment(Segment *seg);
  inline bool hasCamera(CameraType type) const { return camera_states_[type] != nullptr; }
  void pushFrame(CameraType type, std::shared_ptr<Segment> seg, uint32_t segmentId);

private:
  cl_device_id device_id_ = nullptr;
  cl_context context_ = nullptr;
  VisionIpcServer *vipc_server_ = nullptr;
  std::atomic<bool> exit_ = false;
  int segment = -1;

  struct CameraState {
    std::thread thread;
    int width, height;
    VisionStreamType stream_type;
    SafeQueue<std::pair<std::shared_ptr<Segment>, uint32_t>> queue;
  };
  CameraState *camera_states_[MAX_CAMERAS] = {};
  void cameraThread(CameraType cam_type, CameraState *s);
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
  void doMergeEvent();
private:
  QString elapsedTime(uint64_t ns);
  void seekTo(uint64_t to_ts);
  void queueSegmentThread();
  void streamThread();
  std::vector<Event*>::iterator getEvent(uint64_t tm, cereal::Event::Which which);
  std::vector<Event *>::iterator currentEvent();

  void pushFrame(int cur_seg_num, CameraType type, uint32_t frame_id);
  void mergeEvents(Segment *seg);
  std::shared_ptr<Segment> getSegment(int segment);
  const std::string &eventSocketName(const cereal::Event::Reader &e);


  std::atomic<uint64_t> current_ts_ = 0, seek_ts_ = 0;  // ms
  std::atomic<int> current_segment_ = 0;
  std::atomic<uint64_t> route_start_ts_ = 0;
  std::unordered_map<cereal::Event::Which, std::string> eventNameMap;

  // messaging
  SubMaster *sm_ = nullptr;
  PubMaster *pm_ = nullptr;
  std::set<std::string> socks_;

  // segments
  Route route_;
  std::mutex mutex_;
  std::atomic<bool> events_changed_ = false;
  std::map<int, std::shared_ptr<Segment>> segments_;
  std::vector<Event*> *events_;
  // EncodeIdxMap encoderIdx_[MAX_CAMERAS] = {};

  // vipc server
  CameraServer camera_server_;

  std::atomic<bool> exit_ = false;
  std::thread stream_thread_, queue_thread_;
  friend class Segment;
};
