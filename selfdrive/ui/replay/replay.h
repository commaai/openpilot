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
  FrameReader *getFrameReader(const std::string &type) const {
    if (type == "roadCameraState") {
      return road_cam_reader;
    } else {
      return driver_cam_reader;
    }
  }
  LogReader *log_reader = nullptr;
  FrameReader *road_cam_reader = nullptr;
  FrameReader *driver_cam_reader = nullptr;
  std::atomic<int> loading;
};

class Replay : public QObject {
  Q_OBJECT

public:
  Replay(const QString &route, SubMaster *sm = nullptr, QObject *parent = nullptr);
  ~Replay();
  void start();

private:
  void addSegment(int n);
  const SegmentData *getSegment(int n);
  void removeSegment(int n);
  std::optional<std::pair<FrameReader*, int>> getFrame(const std::string &type, uint32_t frame_id);

  void streamThread();
  void keyboardThread();
  void segmentQueueThread();
  void cameraThread();
  
  void parseResponse(const QString &response);
  void seekTime(int ts);
  void startVipcServer(const SegmentData *segment);

  float last_print = 0;
  std::atomic<int> seek_ts = 0;
  std::atomic<int> current_ts = 0;
  std::atomic<int> current_segment = 0;
  std::mutex lock;

  HttpRequest *http;
  QStringList road_camera_paths;
  QStringList qcameras_paths;
  QStringList driver_camera_paths;
  QStringList log_paths;

  // messaging
  SubMaster *sm;
  PubMaster *pm;
  std::set<std::string> socks;
  QString route;

  std::atomic<bool> exit_;
  std::mutex segment_lock;
  QMap<int, SegmentData*> segments;
  SafeQueue<std::pair<std::string, uint32_t>> frame_queue;

  VisionIpcServer *vipc_server = nullptr;

  cl_device_id device_id;
  cl_context context;

};
