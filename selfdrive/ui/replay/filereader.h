#pragma once

#include <thread>

#include <QMultiMap>
#include <QNetworkAccessManager>
#include <QReadWriteLock>
#include <QString>
#include <QVector>
#include <QWidget>

#include <capnp/serialize.h>
#include "cereal/gen/cpp/log.capnp.h"

class FileReader : public QObject {
  Q_OBJECT

public:
  FileReader(const QString &fn, QObject *parent);
  void read();
  void abort();

signals:
  void finished(const QByteArray &dat);
  void failed(const QString &err);

private:
  void startHttpRequest();
  QNetworkReply *reply_ = nullptr;
  QUrl url_;
};

class Event {
public:
  Event(const kj::ArrayPtr<const capnp::word> &amsg) : reader(amsg) {
    words = kj::ArrayPtr<const capnp::word>(amsg.begin(), reader.getEnd());
  }
  inline cereal::Event::Reader event() { return reader.getRoot<cereal::Event>(); }
  inline kj::ArrayPtr<const capnp::byte> bytes() const { return words.asBytes(); }

  kj::ArrayPtr<const capnp::word> words;
  capnp::FlatArrayMessageReader reader;
};

typedef QMultiMap<uint64_t, Event*> Events;

class LogReader : public QObject {
  Q_OBJECT

public:
  LogReader(const QString& file, Events *, QReadWriteLock* events_lock_, QMap<int, QPair<int, int> > *eidx_);
  ~LogReader();

private:
  void start();
  void fileReady(const QByteArray &dat);
  void parseEvents(kj::ArrayPtr<const capnp::word> words);

  FileReader *file_reader_ = nullptr;
  std::vector<uint8_t> raw_;
  Events events_;

  // global
  Events *events;
  QReadWriteLock* events_lock;
  QMap<int, QPair<int, int> > *eidx;

  std::atomic<bool> exit_ = false;
  QThread *thread_ = nullptr;
};
