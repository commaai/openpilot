#pragma once

#include <iostream>
#include <termios.h>

#include <QJsonArray>
#include <QThread>

#include <capnp/dynamic.h>

#include "cereal/visionipc/visionipc_server.h"
#include "selfdrive/common/queue.h"
#include "selfdrive/common/util.h"
#include "selfdrive/ui/qt/api.h"
#include "selfdrive/ui/replay/filereader.h"
#include "selfdrive/ui/replay/framereader.h"

struct SegmentData {
  LogReader *log_reader = nullptr;
  FrameReader *road_cam_reader = nullptr;
  FrameReader *driver_cam_reader = nullptr;
  std::atomic<int> loading;
};

typedef QMap<int, QPair<int, int>> EncodeidxMap;

class Replay : public QObject {
  Q_OBJECT

public:
  Replay(const QString &route, SubMaster *sm = nullptr, QObject *parent = nullptr);
  void start();

private:
  void addSegment(int n);
  const SegmentData *getSegment(int n);
  void removeSegment(int n);

  void streamThread();
  void keyboardThread();
  void segmentQueueThread();
  void cameraThread();
  
  void parseResponse(const QString &response);
  void publishFrame(const std::string &type, const cereal::Event::Reader &event);
  void seekTime(int ts);

  float last_print = 0;
  std::atomic<int> seek_ts = 0;
  std::atomic<int> current_ts = 0;
  std::atomic<int> current_segment = 0;

  // logs
  Events events;
  QReadWriteLock events_lock;
  EncodeidxMap eidx;

  HttpRequest *http;
  QStringList road_camera_paths;
  QStringList qcameras_paths;
  QStringList driver_camera_paths;
  QStringList log_paths;
  
  // messaging
  SubMaster *sm;
  PubMaster *pm;
  QVector<std::string> socks;
  QString route;

  std::atomic<bool> exit_;
  std::mutex segment_lock;
  QMap<int, SegmentData*> segments;
  SafeQueue<std::pair<FrameReader *, int>> frame_queue;

  VisionIpcServer *vipc_server = nullptr;
};
