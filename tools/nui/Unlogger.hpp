#ifndef UNLOGGER_HPP
#define UNLOGGER_HPP

#include <QThread>
#include <QReadWriteLock>
#include "messaging.hpp"
#include "FileReader.hpp"
#include "FrameReader.hpp"

class Unlogger : public QObject {
Q_OBJECT
  public:
    Unlogger(Events *events_, QReadWriteLock* events_lock_, QMap<int, FrameReader*> *frs_, int seek);
    uint64_t getCurrentTime() { return tc; }
    void setSeekRequest(uint64_t seek_request_) { seek_request = seek_request_; }
    void setPause(bool pause) { paused = pause; }
    void togglePause() { paused = !paused; }
    QMap<int, QPair<int, int> > eidx;
  public slots:
    void process();
  signals:
    void elapsed();
    void finished();
  private:
    Events *events;
    QReadWriteLock *events_lock;
    QMap<int, FrameReader*> *frs;
    QMap<int, PubSocket*> socks;
    Context *ctx;
    uint64_t tc = 0;
    uint64_t seek_request = 0;
    bool paused = false;
};

#endif

