#pragma once

#include <iostream>
#include <termios.h>

#include <QFile>
#include <QQueue>
#include <QThread>
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonDocument>

#include <capnp/dynamic.h>

#include "qt/api.hpp"
#include "FileReader.hpp"
#include "FrameReader.hpp"
#include "visionipc_server.h"

#include "clutil.h"
#include "common/util.h"
#include "common/timing.h"
#include "cereal/services.h"

// TODO: figure out why logs don't donwload when this is QObject after removing Unlogger
class Replay : public QWidget {
  Q_OBJECT

public:
  Replay(QString route_);
  void start(SubMaster *sm = nullptr);
  void addSegment(int i);
  void trimSegment(int seg_num);
  void seekTime(int seek_);
  QJsonArray camera_paths;
  QJsonArray log_paths;

  void togglePause() { paused = !paused; }
  uint64_t getCurrentTime() { return tc; }
  uint64_t getRelativeCurrentTime() { return tc - route_t0; }
  void setSeekRequest(uint64_t seek_request_) {
    seeking = true;
    seek_request = seek_request_;
  }

public slots:
  void seekThread();
  void seekRequestThread();
  void parseResponse(QString response);
  void stream(SubMaster *sm = nullptr);

private:
  int seek;
  QString route;
  int current_segment;

  QThread *thread;
  QThread *seek_thread;
  QThread *queue_thread;
  QQueue<QPair<bool, int>> seek_queue;
  int window_padding = 1;

  uint64_t tc = 0;
  float last_print = 0;
  uint64_t route_t0;
  uint64_t seek_request = 0;
  bool paused = false;
  bool seeking = false;
  bool loading_segment = false;

  Events events;
  QReadWriteLock events_lock;
  QMap<int, QPair<int, int> > eidx;

  Context *ctx;
  HttpRequest *http;
  QMap<int, LogReader*> lrs;
  QMap<int, FrameReader*> frs;
  QMap<std::string, PubSocket*> socks;
  VisionIpcServer *vipc_server = nullptr;
};
