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
  void seekTime(int seek_, bool just_update = false);
  QJsonArray camera_paths;
  QJsonArray log_paths;

public slots:
  void parseResponse(QString response);
  void seekThread();

protected:
  Unlogger *unlogger;

private:
  int seek;
  QString route;
  int current_segment;

  QThread *thread;
  QThread *seek_thread;
  int window_padding = 1;

  Events events;
  QReadWriteLock events_lock;

  HttpRequest *http;
  QMap<int, LogReader*> lrs;
  QMap<int, FrameReader*> frs;
};
