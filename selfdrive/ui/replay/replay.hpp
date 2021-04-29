#pragma once

#include <QFile>
#include <QQueue>
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonDocument>

#include <capnp/dynamic.h>

#include "qt/api.hpp"
#include "Unlogger.hpp"
#include "FileReader.hpp"
#include "FrameReader.hpp"
#include "visionipc_server.h"

#include "common/util.h"

class Replay : public QObject {
  Q_OBJECT

public:
  Replay(QString route_, int seek);
  void stream(SubMaster *sm = nullptr);
  void addSegment(int i);
  void trimSegment(int seg_num);
  void seekTime(int seek_);
  QJsonArray camera_paths;
  QJsonArray log_paths;

public slots:
  void parseResponse(QString response);
  void seekThread();

protected:
  Unlogger *unlogger;

private:
  int current_segment;
  QString route;
  int seek;

  QThread *thread;
  QThread *seek_thread;

  QReadWriteLock events_lock;
  Events events;

  QMap<int, LogReader*> lrs;
  QMap<int, FrameReader*> frs;
  HttpRequest *http;
};
