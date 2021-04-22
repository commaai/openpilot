#pragma once

#include <QFile>
#include <QQueue>
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonDocument>

#include <capnp/dynamic.h>

#include "Unlogger.hpp"
#include "FileReader.hpp"
#include "FrameReader.hpp"
#include "visionipc_server.h"

class Replay : public QObject {

public:
  Replay(QString route_, int seek, int use_api);
  void stream(int seek, SubMaster *sm = nullptr);
  bool addSegment(int i);
  void trimSegment(int n);
  QJsonArray camera_paths;
  QJsonArray log_paths;
  int use_api;

  QQueue<int> event_sizes;

protected:
  Unlogger *unlogger;

private:
  QString route;

  QReadWriteLock events_lock;
  Events events;

  QMap<int, LogReader*> lrs;
  QMap<int, FrameReader*> frs;

  int seg_add;
};

