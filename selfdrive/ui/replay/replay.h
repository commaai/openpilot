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


constexpr int FORWARD_SEGS = 2;
constexpr int BACKWARD_SEGS = 2;


class Replay : public QObject {
  Q_OBJECT

public:
  Replay(QString route, SubMaster *sm = nullptr, QObject *parent = 0);

  void start();
  void addSegment(int n);
  void removeSegment(int n);
  void seekTime(int ts);

  void parseResponse(const QString &response);

private:
  void streamThread();
  void keyboardThread();
  void segmentQueueThread();
  void cameraThread();
  void pushFrameToQueue(uint32_t frameId, const QMap<int, FrameReader*>& framesMap, const QMap<int, QPair<int, int>>& frameEidx);

  float last_print = 0;
  uint64_t route_start_ts;
  std::atomic<int> seek_ts = 0;
  std::atomic<int> current_ts = 0;
  std::atomic<int> current_segment = 0;

  QThread *stream_thread, *keyboard_thread, *queue_thread, *camera_thread;

  // logs
  Events events;
  QReadWriteLock events_lock;
  QMap<int, QPair<int, int>> eidx;

  HttpRequest *http;
  QJsonArray road_camera_paths;
  QJsonArray qcameras_paths;
  QJsonArray driver_camera_paths;
  
  QJsonArray log_paths;
  
  QMap<int, LogReader*> lrs;
  QMap<int, FrameReader*> frs;

  // messaging
  SubMaster *sm;
  PubMaster *pm;
  QVector<std::string> socks;
  VisionIpcServer *vipc_server = nullptr;
  QString route;
  std::atomic<bool> exit_;
  SafeQueue<std::pair<FrameReader *, int>> frame_queue;
};
