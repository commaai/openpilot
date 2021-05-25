#include "selfdrive/ui/replay/replay.h"

#include <QJsonDocument>
#include <QJsonObject>

#include "cereal/services.h"
#include "selfdrive/camerad/cameras/camera_common.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/hardware/hw.h"

const int SEGMENT_LENGTH = 60; // 60s
const int FORWARD_SEGS = 2;
const int BACKWARD_SEGS = 2;

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

Replay::Replay(const QString &route, SubMaster *sm_, QObject *parent) : route(route), sm(sm_), QObject(parent) {
  QStringList block = QString(getenv("BLOCK")).split(",");
  qDebug() << "blocklist" << block;

  QStringList allow = QString(getenv("ALLOW")).split(",");
  qDebug() << "allowlist" << allow;

  std::vector<const char*> s;
  for (const auto &it : services) {
    if ((allow[0].size() == 0 || allow.contains(it.name)) &&
        !block.contains(it.name)) {
      s.push_back(it.name);
      socks.append(it.name);
    }
  }
  qDebug() << "services " << s;

  if (sm == nullptr) {
    pm = new PubMaster(s);
  }
}

void Replay::start() {
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

  road_camera_paths = doc["cameras"].toArray();
  qcameras_paths = doc["qcameras"].toArray();
  driver_camera_paths = doc["dcameras"].toArray();
  log_paths = doc["logs"].toArray();

  seekTime(0);

  auto startThread = [=](auto functor) -> QThread * {
    QThread *t = QThread::create(functor, this);
    connect(t, &QThread::finished, t, &QThread::deleteLater);
    t->start();
    return t;
  };

  queue_thread = startThread(&Replay::segmentQueueThread);
  stream_thread = startThread(&Replay::streamThread);
  camera_thread = startThread(&Replay::cameraThread);
  keyboard_thread = startThread(&Replay::keyboardThread);
}

void Replay::cameraThread() {
  bool buffers_initialized[VISION_STREAM_MAX] = {};

  cl_device_id device_id = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT);
  cl_context context = CL_CHECK_ERR(clCreateContext(NULL, 1, &device_id, NULL, NULL, &err));

  vipc_server = new VisionIpcServer("camerad", device_id, context);
  vipc_server->start_listener();

  while (!exit_) {
    std::pair<FrameReader *, int> frame;
    if (!frame_queue.try_pop(frame, 50)) continue;

    auto [frameReader, idx] = frame;
    if (!buffers_initialized[frameReader->stream_type]) {
      vipc_server->create_buffers(frameReader->stream_type, UI_BUF_COUNT, true, frameReader->width, frameReader->height);
      buffers_initialized[frameReader->stream_type] = true;
    }

    VisionIpcBufExtra extra = {};
    VisionBuf *buf = vipc_server->get_buffer(frameReader->stream_type);
    memcpy(buf->addr, frameReader->get(idx), frameReader->getRGBSize());
    vipc_server->send(buf, &extra, false);
  }

  CL_CHECK(clReleaseContext(context));
}

void Replay::addSegment(int n) {
  assert((n >= 0) && (n < log_paths.size()));

  std::unique_lock lk(segment_lock);
  if (segments[n] != nullptr) return;

  SegmentData *segment = new SegmentData{.loading = 1};
  segments[n] = segment;
  lk.unlock();

  QThread *t = new QThread;
  segment->log_reader = new LogReader(log_paths.at(n).toString(), &events, &events_lock, &eidx);
  segment->log_reader->moveToThread(t);
  connect(segment->log_reader, &LogReader::done, [&] { --segment->loading; });
  QObject::connect(t, &QThread::started, segment->log_reader, &LogReader::process);
  t->start();

  auto load_frame = [&](const QString &path, VisionStreamType stream_type) {
    segment->loading += 1;
    FrameReader *frame_reader = new FrameReader(qPrintable(path), VISION_STREAM_RGB_BACK);
    connect(frame_reader, &FrameReader::done, [&] { --segment->loading; });
    QThread *t = QThread::create([=] { frame_reader->process(); });
    QObject::connect(t, &QThread::finished, t, &QThread::deleteLater);
    t->start();
    return frame_reader;
  };

  if (n < road_camera_paths.size()) {
    segment->road_cam_reader = load_frame(road_camera_paths.at(n).toString(), VISION_STREAM_RGB_BACK);
  }
  if (n < driver_camera_paths.size()) {
    segment->driver_cam_reader = load_frame(driver_camera_paths.at(n).toString(), VISION_STREAM_RGB_FRONT);
  }
}

void Replay::removeSegment(int n) {
  std::unique_lock lk(segment_lock);
  if (segments.contains(n)) {
    auto s = segments.take(n);
    delete s->log_reader;
    delete s->road_cam_reader;
    delete s->driver_cam_reader;
    delete s;
  }
}

void Replay::seekTime(int ts) {
  ts = std::clamp(ts, 0, log_paths.size() * SEGMENT_LENGTH);
  qInfo() << "seeking to " << ts;

  seek_ts = ts;
  current_segment = ts / SEGMENT_LENGTH;
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

void Replay::streamThread() {
  QElapsedTimer timer;
  timer.start();

  route_start_ts = 0;
  while (true) {
    SegmentData * segment = nullptr;
    {
      std::unique_lock lk(segment_lock);
      segment = segments[current_segment];
      if (!segment || segment->loading) {
        lk.unlock();
        qDebug() << "waiting for events";
        QThread::msleep(100);
        continue;
      }
    }

    // TODO: use initData's logMonoTime
    if (route_start_ts == 0) {
      route_start_ts = events.firstKey();
    }

    uint64_t t0 = route_start_ts + (seek_ts * 1e9);
    seek_ts = -1;
    qDebug() << "unlogging at" << (t0 - route_start_ts) / 1e9;

    // wait until we have events within 1s of the current time
    auto eit = events.lowerBound(t0);
    while (eit.key() - t0 > 1e9) {
      eit = events.lowerBound(t0);
      QThread::msleep(10);
    }

    uint64_t t0r = timer.nsecsElapsed();
    while ((eit != events.end()) && seek_ts < 0) {
      cereal::Event::Reader e = (*eit);
      std::string type;
      KJ_IF_MAYBE(e_, static_cast<capnp::DynamicStruct::Reader>(e).which()) {
        type = e_->getProto().getName();
      }

      uint64_t tm = e.getLogMonoTime();
      current_ts = std::max(tm - route_start_ts, (unsigned long)0) / 1e9;

      if (socks.contains(type)) {
        if (std::abs(current_ts - last_print) > 5.0) {
          last_print = current_ts;
          qInfo() << "at " << last_print;
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
          if (auto it_ = eidx.find(e.getRoadCameraState().getFrameId()); it_ != eidx.end()) {
            auto [segment, idx] = *it_;
            SegmentData *frame_segment = segments[segment];
            if (frame_segment && !frame_segment->loading) {
              frame_queue.push({frame_segment->road_cam_reader, idx});
            }
          }
        } else if (type == "driverCameraState") {

        }

        // publish msg
        if (sm == nullptr) {
          MessageBuilder msg;
          msg.setRoot(e);
          pm->send(type.c_str(), msg);
        } else {
          sm->update_msgs(nanos_since_boot(), {{type, e}});
        }
      }

      ++eit;
    }
  }
}
