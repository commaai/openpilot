#pragma once

#include <QString>
#include <QNetworkAccessManager>
#include <QWidget>
#include <QVector>
#include <QMultiMap>
#include <QElapsedTimer>
#include <QReadWriteLock>

#include <bzlib.h>

#include <kj/io.h>
#include <capnp/serialize.h>

#include "cereal/gen/cpp/log.capnp.h"

#include <thread>
#include "channel.hpp"

class FileReader : public QObject {
  Q_OBJECT

public:
  FileReader(const QString& file_);
  void startRequest(const QUrl &url);
  ~FileReader();
  virtual void readyRead();
  void httpFinished();
  virtual void done() {};

public slots:
  void process();

protected:
  QNetworkReply *reply;

private:
  QNetworkAccessManager *qnam;
  QElapsedTimer timer;
  QString file;
};

typedef QMultiMap<uint64_t, cereal::Event::Reader> Events;

class LogReader : public FileReader {
Q_OBJECT
public:
  LogReader(const QString& file, Events *, QReadWriteLock* events_lock_, QMap<int, QPair<int, int> > *eidx_);
  ~LogReader();

  void readyRead();
  void done() { is_done = true; };
  bool is_done = false;

private:
  bz_stream bStream;

  // backing store
  QByteArray raw;

  std::thread *parser;
  int event_offset;
  channel<int> cdled;

  // global
  void mergeEvents(int dled);
  Events *events;
  QReadWriteLock* events_lock;
  QMap<int, QPair<int, int> > *eidx;
};
