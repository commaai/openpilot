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

class Segment {
public:
  Segment(int segment_id) : id(segment_id) {}
  ~Segment() {
    qDebug() << QString("remove segment %1").arg(id);
    delete log;
    for (auto f : frames) delete f;
  }

  const int id;
  LogReader *log = nullptr;
  FrameReader *frames[MAX_FRAME_TYPE] = {};
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
  bool loadFromJson(const QString &response);

private:
  void addSegment(int n);
  void removeSegment(int n);
  std::shared_ptr<Segment> getSegment(int n);

  void streamThread();
  void keyboardThread();
  void segmentQueueThread();
  void cameraThread(FrameType frame_type);

  void seekTime(int ts);
  void startVipcServer(const Segment *segment);
  void pushFrame(FrameType type, int seg_id, uint32_t frame_id);

  std::atomic<int64_t> current_ts_ = 0, seek_ts_ = 0;
  std::atomic<int> current_segment_ = 0;

  HttpRequest *http_ = nullptr;
  QString route_;
  QStringList log_paths_;
  QStringList frame_paths_[MAX_FRAME_TYPE];

  // messaging
  SubMaster *sm_ = nullptr;
  PubMaster *pm_ = nullptr;
  std::set<std::string> socks_;

  // segments
  std::mutex segment_lock_;
  std::map<int, std::shared_ptr<Segment>> segments_;
  
  // vipc server
  cl_device_id device_id_;
  cl_context context_;
  VisionIpcServer *vipc_server_ = nullptr;
  
  struct Camera {
    QThread *thread = nullptr;
    SafeQueue<const EncodeIdx*> queue; // <segment_id, frame_id>
  };
  Camera *cameras_[MAX_FRAME_TYPE] = {};

  // TODO: quit replay gracefully
  std::atomic<bool> exit_ = false;

};
