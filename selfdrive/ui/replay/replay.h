#pragma once

#include <set>

#include "cereal/visionipc/visionipc_server.h"
#include "selfdrive/common/queue.h"
#include "selfdrive/common/util.h"
#include "selfdrive/ui/replay/filereader.h"
#include "selfdrive/ui/replay/framereader.h"
#include "selfdrive/ui/replay/route.h"

class Segment {
 public:
  Segment(int seg_num, const SegmentFile &file);
  ~Segment();

  const int seg_num;
  LogReader *log = nullptr;
  FrameReader *frames[MAX_CAMERAS] = {};
  std::atomic<bool> loaded = false;

 private:
  std::atomic<int> loading = 0;
};

class CameraServer {
 public:
  CameraServer();
  ~CameraServer();
  void stop();
  void ensureServerForSegment(Segment *seg);
  inline bool hasCamera(CameraType type) const { return camera_states_[type] != nullptr; }
  inline void pushFrame(CameraType type, std::shared_ptr<Segment> seg, uint32_t segmentId) {
    camera_states_[type]->queue.push({seg, segmentId});
  }

 private:
  cl_device_id device_id_ = nullptr;
  cl_context context_ = nullptr;
  VisionIpcServer *vipc_server_ = nullptr;
  std::atomic<bool> exit_ = false;

  struct CameraState {
    std::thread thread;
    int width, height;
    VisionStreamType stream_type;
    SafeQueue<std::pair<std::shared_ptr<Segment>, uint32_t>> queue;
  };
  CameraState *camera_states_[MAX_CAMERAS] = {};
  void cameraThread(CameraType cam_type, CameraState *s);
};

class Replay {
 public:
  Replay(SubMaster *sm = nullptr);
  ~Replay();
  bool start(const QString &routeName);
  bool start(const Route &route);
  void seekTo(int to_ts);
  void relativeSeek(int ts);
  void stop();
  bool running() { return stream_thread_.joinable(); }

 private:
  std::shared_ptr<Segment> getSegment(int segment);
  void queueSegment(int segment);
  const std::string &eventName(const cereal::Event::Reader &e);

  void streamThread();
  void pushFrame(CameraType type, int seg_id, uint32_t frame_id);

  std::atomic<int64_t> current_ts_ = 0, seek_ts_ = 0;
  std::atomic<int> current_segment_ = -1;
  std::unordered_map<cereal::Event::Which, std::string> eventNameMap;

  // messaging
  SubMaster *sm_ = nullptr;
  PubMaster *pm_ = nullptr;
  std::set<std::string> socks_;

  // segments
  Route route_;
  std::map<int, std::shared_ptr<Segment>> segments_;

  // vipc server
  CameraServer camera_server_;

  std::atomic<bool> exit_ = false;
  std::thread stream_thread_;
};
