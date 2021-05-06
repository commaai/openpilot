#pragma once

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QQueue>

#include <capnp/dynamic.h>

#include "cereal/visionipc/visionipc_server.h"
#include "selfdrive/common/util.h"
#include "selfdrive/ui/qt/api.h"
#include "selfdrive/ui/replay/filereader.h"
#include "selfdrive/ui/replay/unlogger.h"
#include "tools/clib/framereader.h"

class Replay : public QObject {
  Q_OBJECT

public:
  Replay(QString route_, int seek);
  void stream(SubMaster *sm = nullptr);
  void addSegment(int i);
  QJsonArray camera_paths;
  QJsonArray log_paths;

  QQueue<int> event_sizes;

public slots:
  void parseResponse(const QString &response);

protected:
  Unlogger *unlogger;

private:
  QString route;

  QReadWriteLock events_lock;
  Events events;

  QMap<int, LogReader*> lrs;
  QMap<int, FrameReader*> frs;
  HttpRequest *http;

  int current_segment;
};

