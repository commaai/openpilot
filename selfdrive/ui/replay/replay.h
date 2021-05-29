#pragma once

#include <iostream>
#include <termios.h>
#include <set>
#include <mutex>

#include <QJsonArray>
#include <QReadWriteLock>
#include <QThread>

#include <capnp/dynamic.h>

#include "cereal/visionipc/visionipc_server.h"
#include "selfdrive/common/queue.h"
#include "selfdrive/common/util.h"
#include "selfdrive/ui/qt/api.h"
#include "selfdrive/ui/replay/filereader.h"
#include "selfdrive/ui/replay/framereader.h"

struct SegmentFiles {
  QString log_file;
  QString cam_file;
  QString dcam_file;
  QString wcam_file;
  QString qcam_files;
};

class Segment {
 public:
  Segment(int segment_id, const SegmentFiles &files);
  ~Segment();

 public:
  const int id;
  LogReader *log = nullptr;
  FrameReader *frames[MAX_CAMERAS] = {};
  std::atomic<int> loading;
};

class Replay : public QObject {
  Q_OBJECT

 public:
  Replay(const QString &route, SubMaster *sm = nullptr, QObject *parent = nullptr);
  ~Replay();
  void load();
  bool loadFromLocal();
  void loadFromServer();
  bool loadFromJson(const QString &json);
  bool loadSegments(const QMap<int, QMap<QString, QString>> &segment_paths);
  void clear();

 private:
  std::shared_ptr<Segment> getSegment(int n);

  void streamThread();
  void keyboardThread();
  void segmentQueueThread();
  void cameraThread(CameraType cam_type, VisionStreamType stream_type);

  void seekTime(int ts);
  void startVipcServer(const Segment *segment);
  void pushFrame(CameraType type, int seg_id, uint32_t frame_id);

  std::atomic<int64_t> current_ts_ = 0, seek_ts_ = 0;
  std::atomic<int> current_segment_ = 0;

  QString route_;

  // messaging
  SubMaster *sm_ = nullptr;
  PubMaster *pm_ = nullptr;
  std::set<std::string> socks_;

  // segments
  std::mutex segment_lock_;
  std::map<int, std::shared_ptr<Segment>> segments_;
  QMap<int, SegmentFiles> segment_paths_;

  // vipc server
  cl_device_id device_id_;
  cl_context context_;
  VisionIpcServer *vipc_server_ = nullptr;
  SafeQueue<const EncodeIdx *> *frame_queues_[MAX_CAMERAS] = {};

  // TODO: quit replay gracefully
  std::atomic<bool> exit_ = false;
};
