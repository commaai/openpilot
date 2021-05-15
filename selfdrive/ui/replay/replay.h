#pragma once

#include <iostream>
#include <termios.h>

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QThread>
#include <QQueue>

#include <capnp/dynamic.h>

#include "cereal/visionipc/visionipc_server.h"
#include "selfdrive/common/util.h"
#include "selfdrive/ui/qt/api.h"
#include "selfdrive/ui/replay/filereader.h"
#include "tools/clib/framereader.h"


constexpr int WINDOW_SIZE = 2;


class Replay : public QObject {
  Q_OBJECT

public:
  Replay(QString route_, SubMaster *sm = nullptr, QObject *parent = 0);
  void start();
  void addSegment(int i);
  void trimSegment(int seg_num);
  void seekTime(int seek_);

  uint64_t getCurrentTime() { return tc; }
  uint64_t getRelativeCurrentTime() { return tc - route_start_ts; }

public slots:
  void keyboardThread();
  void seekRequestThread();
  void parseResponse(const QString &response);
  void stream();

private:
  QString route;
  int current_segment;

  QThread *thread;
  QThread *kb_thread;
  QThread *queue_thread;
  QQueue<QPair<bool, int>> seek_queue;
  int window_padding = 1;

  uint64_t tc = 0;
  float last_print = 0;
  uint64_t route_start_ts;
  bool seeking = false;

  Events events;
  QReadWriteLock events_lock;
  QMap<int, QPair<int, int>> eidx;

  HttpRequest *http;
  QJsonArray camera_paths;
  QJsonArray log_paths;
  QMap<int, LogReader*> lrs;
  QMap<int, FrameReader*> frs;

  // messaging
  Context *ctx;
  SubMaster *sm;
  QMap<std::string, PubSocket*> socks;
  VisionIpcServer *vipc_server = nullptr;
};
