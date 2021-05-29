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

class EventMsg {
  public:
  EventMsg(const kj::ArrayPtr<const capnp::word> &amsg) : msg(amsg) {
    words = kj::ArrayPtr<const capnp::word>(amsg.begin(), msg.getEnd());
  }
  kj::ArrayPtr<const capnp::word> words;
  capnp::FlatArrayMessageReader msg;
};

typedef QMultiMap<uint64_t, EventMsg*> Events;

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
