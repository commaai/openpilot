#pragma once

#include <set>
#include <mutex>

#include <QReadWriteLock>
#include <QThread>

#include <capnp/dynamic.h>

#include "cereal/visionipc/visionipc_server.h"
#include "selfdrive/common/queue.h"
#include "selfdrive/common/util.h"

#include "selfdrive/ui/replay/route.h"
#include "selfdrive/ui/replay/filereader.h"
#include "selfdrive/ui/replay/framereader.h"

class Segment {
 public:
  Segment(int seg_num, const SegmentFiles &files);
  ~Segment();

  const int seg_num;
  LogReader *log = nullptr;
  FrameReader *frames[MAX_CAMERAS] = {};
  std::atomic<bool> loaded = false;
  std::atomic<int> loading = 0;
};

class CameraServer {
public:
  CameraServer();
  ~CameraServer();
  void stop();
  void cameraThread(CameraType cam_type, VisionStreamType stream_type);
  void ensureServer(Segment *seg);
  bool hasCamera(CameraType type) const { return frame_queues_[type] != nullptr; }
  void pushFrame(CameraType type, std::shared_ptr<Segment> seg, uint32_t segmentId) { frame_queues_[type]->push({seg, segmentId}); }
  cl_device_id device_id_ = nullptr;
  cl_context context_ = nullptr;
  VisionIpcServer *vipc_server_ = nullptr;
  int road_cam_width_ = 0, road_cam_height_ = 0;
  SafeQueue<std::pair<std::shared_ptr<Segment>, uint32_t>> *frame_queues_[MAX_CAMERAS] = {};
  std::atomic<bool> exit_ = false;
  std::vector<std::thread> threads_;
};

class Replay : public QObject {
  Q_OBJECT

public:
  Replay(SubMaster *sm = nullptr, QObject *parent = nullptr);
  ~Replay();
  bool load(const QString &routeName);
  bool load(const Route &route);
  void seekTo(int to_ts);
  void relativeSeek(int ts);
  void clear();

private:
  std::shared_ptr<Segment> getSegment(int n);

  void streamThread();
  void segmentQueueThread();
  void cameraThread(CameraType cam_type, VisionStreamType stream_type);
  
  void pushFrame(CameraType type, int seg_id, uint32_t frame_id);

  std::atomic<int64_t> current_ts_ = 0, seek_ts_ = 0;
  std::atomic<int> current_segment_ = 0;

  // messaging
  SubMaster *sm_ = nullptr;
  PubMaster *pm_ = nullptr;
  std::set<std::string> socks_;

  // segments
  Route route_;
  std::mutex segment_lock_;
  std::map<int, std::shared_ptr<Segment>> segments_;
  
  // vipc server
  CameraServer camera_server_;

  // TODO: quit replay gracefully
  std::atomic<bool> exit_ = false;
};
