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
      socks.insert(it.name);
    }
  }
  qDebug() << "services " << s;

  if (sm == nullptr) {
    pm = new PubMaster(s);
  }

  device_id = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT);
  context = CL_CHECK_ERR(clCreateContext(NULL, 1, &device_id, NULL, NULL, &err));
}

Replay::~Replay() {
  CL_CHECK(clReleaseContext(context));
}

void Replay::start() {
  const QString url = "https://api.commadotai.com/v1/route/" + route + "/files";
  http = new HttpRequest(this, url, "", !Hardware::PC());
  QObject::connect(http, &HttpRequest::receivedResponse, this, &Replay::loadJson);
}

void Replay::loadJson(const QString &json) {
  QJsonDocument doc = QJsonDocument::fromJson(json.trimmed().toUtf8());
  if (doc.isNull()) {
    qDebug() << "JSON Parse failed";
    return;
  }

  road_camera_paths = doc["cameras"].toVariant().toStringList();
  qcameras_paths = doc["qcameras"].toVariant().toStringList();
  driver_camera_paths = doc["dcameras"].toVariant().toStringList();
  log_paths = doc["logs"].toVariant().toStringList();

  typedef void (Replay::*threadFunc)();
  threadFunc threads[] = {&Replay::segmentQueueThread,
                          &Replay::keyboardThread,
                          &Replay::streamThread};
  for (auto func : threads) {
    QThread *t = QThread::create(func, this);
    connect(t, &QThread::finished, t, &QThread::deleteLater);
    t->start();
  }
}

void Replay::addSegment(int n) {
  assert((n >= 0) && (n < log_paths.size()));

  std::unique_lock lk(segment_lock);
  if (segments[n] != nullptr) return;

  SegmentData *segment = new SegmentData;
  segment->loading = 1;
  segments[n] = segment;
  lk.unlock();

  QThread *t = new QThread;
  segment->log_reader = new LogReader(log_paths[n]);
  segment->log_reader->moveToThread(t);
  connect(segment->log_reader, &LogReader::done, [&] { 
    --segment->loading; 
    t->quit();
  });
  QObject::connect(t, &QThread::started, segment->log_reader, &LogReader::process);
  QObject::connect(t, &QThread::finished, t, &QThread::deleteLater);
  t->start();

  auto read_frames = [&](const QString &path, VisionStreamType stream_type) {
    segment->loading += 1;
    FrameReader *frame_reader = new FrameReader(qPrintable(path), VISION_STREAM_RGB_BACK);
    connect(frame_reader, &FrameReader::done, [&] { --segment->loading; });
    QThread *t = QThread::create([=] { frame_reader->process(); });
    QObject::connect(t, &QThread::finished, t, &QThread::deleteLater);
    t->start();
    return frame_reader;
  };

  if (n < road_camera_paths.size()) {
    segment->road_cam_reader = read_frames(road_camera_paths[n], VISION_STREAM_RGB_BACK);
  }
  if (n < driver_camera_paths.size()) {
    segment->driver_cam_reader = read_frames(driver_camera_paths[n], VISION_STREAM_RGB_FRONT);
  }
}

const SegmentData *Replay::getSegment(int n) {
  std::unique_lock lk(segment_lock);
  const SegmentData *segment = segments[n];
  return (segment && !segment->loading) ? segment : nullptr;
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

void Replay::startVipcServer(const SegmentData *segment) {
  assert(vipc_server == nullptr);

  FrameReader *frames[] = {segment->road_cam_reader, segment->wide_road_cam_reader, segment->driver_cam_reader};
  bool hasFrames = false;
  for (auto f : frames) {
    if (f && f->valid()) hasFrames = true;
  }
  if (hasFrames) {
    vipc_server = new VisionIpcServer("camerad", device_id, context);
    for (auto f : frames) {
      if (f && f->valid()) {
        vipc_server->create_buffers(f->stream_type, UI_BUF_COUNT, true, f->width, f->height);
      }
    }
    QThread *t = QThread::create(&Replay::cameraThread, this);
    connect(t, &QThread::finished, t, &QThread::deleteLater);
    t->start();
    qDebug() << "start vipc server";
  }
}


std::optional<std::pair<FrameReader*, int>> Replay::getFrame(const std::string &type, uint32_t frame_id) {
  std::unique_lock lk(segment_lock);
  auto frame = segments[current_segment]->log_reader->getFrameEncodeIdx(type, frame_id);
  if (!frame) {
    // search in adjacent segments
    int start = std::max(current_segment - BACKWARD_SEGS, 0);
    int end = std::min(current_segment + FORWARD_SEGS, segments.size());
    for (int i = start; i < end; ++i) {
      if (i != current_segment) {
        const SegmentData *segment = segments[i];
        if (segment && !segment->loading) {
          frame = segment->log_reader->getFrameEncodeIdx(type, frame_id);
          if (frame) break;
        }
      }
    }
  }
  if (frame) {
    auto [segment_id, idx] = *frame;
    const SegmentData *segment = segments[segment_id];
    if (segment && !segment->loading) {
      FrameReader *reader = segment->getFrameReader(type);
      if (reader) {
        return std::make_pair(reader, idx);
      }
    }
  }
  return std::nullopt;
}

// threads

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

void Replay::cameraThread() {
  vipc_server->start_listener();
  while (!exit_) {
    std::pair<std::string, uint32_t> frame;
    if (!frame_queue.try_pop(frame, 50)) continue;

    auto [type, frame_id] = frame;
    if (auto f = getFrame(type, frame_id); f) {
      auto [frameReader, idx] = *f;
      VisionIpcBufExtra extra = {};
      VisionBuf *buf = vipc_server->get_buffer(frameReader->stream_type);
      memcpy(buf->addr, frameReader->get(idx), frameReader->getRGBSize());
      vipc_server->send(buf, &extra, false);
    }
  }
  delete vipc_server;
  vipc_server = nullptr;
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

  seekTime(0);
  uint64_t route_start_ts = 0;

  while (true) {
    const SegmentData * segment = getSegment(current_segment);
    if (!segment) {
      qDebug() << "waiting for events";
      QThread::msleep(100);
      continue;
    }
    if (vipc_server == nullptr) {
      startVipcServer(segment);
    }
    const Events &events = segment->log_reader->events();

    // TODO: use initData's logMonoTime
    if (route_start_ts == 0) {
      route_start_ts = events.firstKey();
    }

    uint64_t t0 = route_start_ts + (seek_ts * 1e9);
    qDebug() << "unlogging at" << (t0 - route_start_ts) / 1e9;

    auto eit = events.lowerBound(t0);
    assert(eit != events.end());
    uint64_t t0r = timer.nsecsElapsed();
    int current_seek_ts = seek_ts;
    while (current_seek_ts == seek_ts && eit != events.end()) {
      cereal::Event::Reader e = (*eit)->getRoot<cereal::Event>();
      std::string type;
      KJ_IF_MAYBE(e_, static_cast<capnp::DynamicStruct::Reader>(e).which()) {
        type = e_->getProto().getName();
      }

      uint64_t tm = e.getLogMonoTime();
      current_ts = std::max(tm - route_start_ts, (unsigned long)0) / 1e9;

      if (socks.find(type) != socks.end()) {
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
        if (type == "roadCameraState") {
          frame_queue.push({type, e.getRoadCameraState().getFrameId()});
        } else if (type == "driverCameraState") {
          frame_queue.push({type, e.getDriverCameraState().getFrameId()});
         } else if (type == "wideRoadCameraState") {
          frame_queue.push({type, e.getWideRoadCameraState().getFrameId()});
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

    if (current_seek_ts == seek_ts) {
      // move to the next segment
      current_segment += 1;
      seek_ts = current_ts.load();
    }
  }
}
