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
  SegmentData() = default;
  ~SegmentData() {
    delete log;
    for (auto f : frames) delete f;
  }

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
  const SegmentData *getSegment(int n);
  void removeSegment(int n);

  void streamThread();
  void keyboardThread();
  void segmentQueueThread();
  void cameraThread();
  
  void seekTime(int ts);
  void startVipcServer(const SegmentData *segment);
  std::optional<std::pair<FrameReader *, uint32_t>> getFrame(int seg_id, FrameType type, uint32_t frame_id);

  float last_print = 0;
  std::atomic<int> current_ts = 0, seek_ts = 0;
  std::atomic<int> current_segment = 0, playing_segment = 0;
  std::mutex lock;

  HttpRequest *http = nullptr;
  QString route;
  QStringList log_paths;
  QStringList frame_paths[MAX_FRAME_TYPE];

  // messaging
  SubMaster *sm = nullptr;
  PubMaster *pm = nullptr;
  std::set<std::string> socks;

  std::mutex segment_lock;
  QMap<int, SegmentData*> segments;
  SafeQueue<std::pair<FrameType, uint32_t>> frame_queue;

  VisionIpcServer *vipc_server = nullptr;

  cl_device_id device_id;
  cl_context context;

  // TODO: quit replay gracefully
  std::atomic<bool> exit_ = false;

};
