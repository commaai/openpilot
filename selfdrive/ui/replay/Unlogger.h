#pragma once

#include <QThread>
#include <QReadWriteLock>
#include "clutil.h"
#include "messaging.h"
#include "FileReader.h"
#include "FrameReader.h"
#include "visionipc_server.h"

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
    void process(SubMaster *sm = nullptr);
  signals:
    void elapsed();
    void finished();
    void loadSegment();
  private:
    Events *events;
    QReadWriteLock *events_lock;
    QMap<int, FrameReader*> *frs;
    QMap<std::string, PubSocket*> socks;
    Context *ctx;
    uint64_t tc = 0;
    float last_print = 0;
    uint64_t seek_request = 0;
    bool paused = false;
    bool loading_segment = false;

    VisionIpcServer *vipc_server = nullptr;
};

