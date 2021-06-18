#include "selfdrive/ui/replay/replay.h"

#include <QJsonDocument>
#include <QJsonObject>

#include "cereal/services.h"
#include "selfdrive/camerad/cameras/camera_common.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/hardware/hw.h"

int getch() {
  int ch;
  struct termios oldt;
  struct termios newt;

  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);

  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  ch = getchar();
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);

  return ch;
}

Replay::Replay(QString route, SubMaster *sm_, QObject *parent) : sm(sm_), QObject(parent) {
  QStringList block = QString(getenv("BLOCK")).split(",");
  qDebug() << "blocklist" << block;

  QStringList allow = QString(getenv("ALLOW")).split(",");
  qDebug() << "allowlist" << allow;

  std::vector<const char*> s;
  for (const auto &it : services) {
    if ((allow[0].size() == 0 || allow.contains(it.name)) &&
        !block.contains(it.name)) {
      s.push_back(it.name);
      socks.append(std::string(it.name));
    }
  }
  qDebug() << "services " << s;

  if (sm == nullptr) {
    pm = new PubMaster(s);
  }

  const QString url = "https://api.commadotai.com/v1/route/" + route + "/files";
  http = new HttpRequest(this, url, "", !Hardware::PC());
  QObject::connect(http, &HttpRequest::receivedResponse, this, &Replay::parseResponse);
}

void Replay::parseResponse(const QString &response) {
  QJsonDocument doc = QJsonDocument::fromJson(response.trimmed().toUtf8());
  if (doc.isNull()) {
    qDebug() << "JSON Parse failed";
    return;
  }

  camera_paths = doc["cameras"].toArray();
  log_paths = doc["logs"].toArray();

  seekTime(0);
}

void Replay::addSegment(int n) {
  assert((n >= 0) && (n < log_paths.size()) && (n < camera_paths.size()));
  if (lrs.find(n) != lrs.end()) {
    return;
  }

  lrs[n] = new LogReader(log_paths.at(n).toString());
  // this is a queued connection, mergeEvents is executed in the main thread.
  QObject::connect(lrs[n], &LogReader::finished, this, &Replay::mergeEvents);

  frs[n] = new FrameReader(qPrintable(camera_paths.at(n).toString()));
}

void Replay::removeSegment(int n) {
  // TODO: fix LogReader destructors
  /*
  if (lrs.contains(n)) {
    auto lr = lrs.take(n);
    delete lr;
  }

  events_lock.lockForWrite();
  auto eit = events.begin();
  while (eit != events.end()) {
    if(std::abs(eit.key()/1e9 - getCurrentTime()/1e9) > 60.0){
      eit = events.erase(eit);
      continue;
    }
    eit++;
  }
  events_lock.unlock();
  */
  if (frs.contains(n)) {
    auto fr = frs.take(n);
    delete fr;
  }
}

void Replay::mergeEvents() {
  LogReader *log = qobject_cast<LogReader *>(sender());
  events += log->events;
  for (CameraType cam_type : ALL_CAMERAS) {
    eidx[cam_type].merge(log->eidx[cam_type]);
  }
}

void Replay::start(){
  thread = new QThread;
  QObject::connect(thread, &QThread::started, [=](){
    stream();
  });
  thread->start();

  kb_thread = new QThread;
  QObject::connect(kb_thread, &QThread::started, [=](){
    keyboardThread();
  });
  kb_thread->start();

  queue_thread = new QThread;
  QObject::connect(queue_thread, &QThread::started, [=](){
    segmentQueueThread();
  });
  queue_thread->start();
}

void Replay::seekTime(int ts) {
  ts = std::clamp(ts, 0, log_paths.size() * 60);
  qInfo() << "seeking to " << ts;

  seek_ts = ts;
  current_segment = ts/60;
}

void Replay::segmentQueueThread() {
  // maintain the segment window
  while (true) {
    for (int i = 0; i < log_paths.size(); i++) {
      int start_idx = std::max(current_segment - BACKWARD_SEGS, 0);
      int end_idx = std::min(current_segment + FORWARD_SEGS, log_paths.size());
      if (i >= start_idx && i <= end_idx) {
        addSegment(i);
      } else {
        removeSegment(i);
      }
    }
    QThread::msleep(100);
  }
}

void Replay::keyboardThread() {
  char c;
  while (true) {
    c = getch();
    if(c == '\n'){
      printf("Enter seek request: ");
      std::string r;
      std::cin >> r;

      try {
        if(r[0] == '#') {
          r.erase(0, 1);
          seekTime(std::stoi(r)*60);
        } else {
          seekTime(std::stoi(r));
        }
      } catch (std::invalid_argument) {
        qDebug() << "invalid argument";
      }
      getch(); // remove \n from entering seek
    } else if (c == 'm') {
      seekTime(current_ts + 60);
    } else if (c == 'M') {
      seekTime(current_ts - 60);
    } else if (c == 's') {
      seekTime(current_ts + 10);
    } else if (c == 'S') {
      seekTime(current_ts - 10);
    } else if (c == 'G') {
      seekTime(0);
    }
  }
}

void Replay::stream() {
  QElapsedTimer timer;
  timer.start();

  route_start_ts = 0;
  while (true) {
    if (events.size() == 0) {
      qDebug() << "waiting for events";
      QThread::msleep(100);
      continue;
    }

    // TODO: use initData's logMonoTime
    if (route_start_ts == 0) {
      route_start_ts = events.firstKey();
    }

    uint64_t t0 = route_start_ts + (seek_ts * 1e9);
    seek_ts = -1;
    qDebug() << "unlogging at" << int((t0 - route_start_ts) / 1e9);

    // wait until we have events within 1s of the current time
    auto eit = events.lowerBound(t0);
    while (eit.key() - t0 > 1e9) {
      eit = events.lowerBound(t0);
      QThread::msleep(10);
    }

    uint64_t t0r = timer.nsecsElapsed();
    while ((eit != events.end()) && seek_ts < 0) {
      cereal::Event::Reader e = (*eit)->event;
      std::string type;
      KJ_IF_MAYBE(e_, static_cast<capnp::DynamicStruct::Reader>(e).which()) {
        type = e_->getProto().getName();
      }

      uint64_t tm = e.getLogMonoTime();
      current_ts = std::max(tm - route_start_ts, (unsigned long)0) / 1e9;

      if (socks.contains(type)) {
        float timestamp = (tm - route_start_ts)/1e9;
        if (std::abs(timestamp - last_print) > 5.0) {
          last_print = timestamp;
          qInfo() << "at " << int(last_print) << "s";
        }

        // keep time
        long etime = tm-t0;
        long rtime = timer.nsecsElapsed() - t0r;
        long us_behind = ((etime-rtime)*1e-3)+0.5;
        if (us_behind > 0 && us_behind < 1e6) {
          QThread::usleep(us_behind);
          //qDebug() << "sleeping" << us_behind << etime << timer.nsecsElapsed();
        }

        // publish frame
        // TODO: publish all frames
        if (type == "roadCameraState") {
          auto fr = e.getRoadCameraState();

          auto it_ = eidx[RoadCam].find(fr.getFrameId());
          if (it_ != eidx[RoadCam].end()) {
            EncodeIdx &e = it_->second;
            if (frs.find(e.segmentNum) != frs.end()) {
              auto frm = frs[e.segmentNum];
              if (vipc_server == nullptr) {
                cl_device_id device_id = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT);
                cl_context context = CL_CHECK_ERR(clCreateContext(NULL, 1, &device_id, NULL, NULL, &err));

                vipc_server = new VisionIpcServer("camerad", device_id, context);
                vipc_server->create_buffers(VisionStreamType::VISION_STREAM_RGB_BACK, UI_BUF_COUNT,
                                            true, frm->width, frm->height);
                vipc_server->start_listener();
              }

              uint8_t *dat = frm->get(e.frameEncodeId);
              if (dat) {
                VisionIpcBufExtra extra = {};
                VisionBuf *buf = vipc_server->get_buffer(VisionStreamType::VISION_STREAM_RGB_BACK);
                memcpy(buf->addr, dat, frm->getRGBSize());
                vipc_server->send(buf, &extra, false);
              }
            }
          }
        }

        // publish msg
        if (sm == nullptr) {
          auto bytes = (*eit)->bytes();
          pm->send(type.c_str(), (capnp::byte *)bytes.begin(), bytes.size());
        } else {
          std::vector<std::pair<std::string, cereal::Event::Reader>> messages;
          messages.push_back({type, e});
          sm->update_msgs(nanos_since_boot(), messages);
        }
      }

      ++eit;
    }
  }
}
