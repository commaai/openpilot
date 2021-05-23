#pragma once

#include <thread>

#include <QElapsedTimer>
#include <QMultiMap>
#include <QNetworkAccessManager>
#include <QReadWriteLock>
#include <QString>
#include <QVector>
#include <QWidget>

#include <bzlib.h>
#include <capnp/serialize.h>
#include <kj/io.h>
#include "cereal/gen/cpp/log.capnp.h"

#include "tools/clib/channel.h"

class FileReader : public QObject {
  Q_OBJECT

public:
  FileReader(const QString& file_);
  void startRequest(const QUrl &url);
  virtual void readyRead();
  void httpFinished();
  virtual void done() {}

public slots:
  void process();

protected:
  QNetworkReply *reply;

private:
  QNetworkAccessManager *qnam;
  QElapsedTimer timer;
  QString file;
};

typedef QMultiMap<uint64_t, capnp::FlatArrayMessageReader*> Events;

class LogReader : public FileReader {
  Q_OBJECT

public:
  LogReader(const QString &file, Events *, QReadWriteLock* events_lock_, QMap<int, QPair<int, int> > *eidx_);
  ~LogReader();
  void readyRead();
  void done() { is_done = true; };
  bool is_done = false;

private:
  void mergeEvents(kj::ArrayPtr<const capnp::word> amsg);

  bz_stream bStream = {};
  // backing store
  QByteArray raw;

  Events *events;
  QReadWriteLock* events_lock;
  QMap<int, QPair<int, int> > *eidx;
};
