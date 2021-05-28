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

class SegmentData {
public:
  SegmentData(int segment_id) : id(segment_id) {}
  ~SegmentData() {
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
  void loadFromServer();
  bool loadFromLocal();
  bool loadFromJson(const QString &response);

private:
  void addSegment(int n);
  std::shared_ptr<SegmentData> getSegment(int n);
  void removeSegment(int n);

  void streamThread();
  void keyboardThread();
  void segmentQueueThread();
  void cameraThread();
  
  void seekTime(int ts);
  void startVipcServer(const SegmentData *segment);
  std::optional<std::pair<std::shared_ptr<SegmentData>, uint32_t>> getFrameSegment(int seg_id, FrameType type, uint32_t frame_id);

  float last_print_ = 0;
  std::atomic<int> current_ts_ = 0, seek_ts_ = 0;
  std::atomic<int> current_segment_ = 0;
  std::mutex lock_;

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
  std::map<int, std::shared_ptr<SegmentData>> segments_;
  
  // vipc server
  cl_device_id device_id_;
  cl_context context_;
  VisionIpcServer *vipc_server_ = nullptr;
  SafeQueue<std::pair<FrameType, uint32_t>> frame_queue_;

  // TODO: quit replay gracefully
  std::atomic<bool> exit_ = false;

};
