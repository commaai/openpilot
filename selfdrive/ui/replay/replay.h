#pragma once

#include <QFile>
#include <QQueue>
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonDocument>

#include <capnp/dynamic.h>

#include "selfdrive/ui/qt/api.h"
#include "selfdrive/ui/replay/Unlogger.h"
#include "selfdrive/ui/replay/FileReader.h"
#include "tools/clib/FrameReader.h"
#include "cereal/visionipc/visionipc_server.h"

#include "selfdrive/common/util.h"

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

