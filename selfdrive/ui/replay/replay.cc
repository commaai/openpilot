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
  for (auto seg : segments) {
    delete seg;
  }
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

  rd_frm_paths = doc["cameras"].toVariant().toStringList();
  qcameras_paths = doc["qcameras"].toVariant().toStringList();
  drv_frm_paths = doc["dcameras"].toVariant().toStringList();
  log_paths = doc["logs"].toVariant().toStringList();

  typedef void (Replay::*threadFunc)();
  threadFunc threads[] = {&Replay::segmentQueueThread, &Replay::keyboardThread, &Replay::streamThread};
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

  SegmentData *seg = new SegmentData;
  seg->loading = 1;
  segments[n] = seg;
  lk.unlock();

  // read log
  QThread *t = new QThread;
  seg->log = new LogReader(log_paths[n]);
  seg->log->moveToThread(t);
  connect(seg->log, &LogReader::done, [&] { 
    --seg->loading; 
    t->quit();
  });
  QObject::connect(t, &QThread::started, seg->log, &LogReader::process);
  QObject::connect(t, &QThread::finished, t, &QThread::deleteLater);
  t->start();

  // read frames
  
  auto read_frames = [&](const QString &path, VisionStreamType stream_type) {
    seg->loading += 1;
    FrameReader *reader = new FrameReader(path.toStdString(), VISION_STREAM_RGB_BACK);
    connect(reader, &FrameReader::done, [&] { --seg->loading; });
    QThread *t = QThread::create([=] { reader->process(); });
    QObject::connect(t, &QThread::finished, t, &QThread::deleteLater);
    t->start();
    return reader;
  };

  if (n < rd_frm_paths.size()) {
    seg->frames[RoadCamFrame] = read_frames(rd_frm_paths[n], VISION_STREAM_RGB_BACK);
  }
  if (n < drv_frm_paths.size()) {
    seg->frames[DriverCamFrame] = read_frames(drv_frm_paths[n], VISION_STREAM_RGB_FRONT);
  }
}

const SegmentData *Replay::getSegment(int n) {
  std::unique_lock lk(segment_lock);
  const SegmentData *seg = segments[n];
  return (seg && !seg->loading) ? seg : nullptr;
}

void Replay::removeSegment(int n) {
  std::unique_lock lk(segment_lock);
  if (segments.contains(n)) {
    delete segments.take(n);
  }
}

void Replay::startVipcServer(const SegmentData *seg) {
  for (auto f : seg->frames) {
    if (f && f->valid()) {
      if (!vipc_server) {
        vipc_server = new VisionIpcServer("camerad", device_id, context);
      }
      vipc_server->create_buffers(f->stream_type, UI_BUF_COUNT, true, f->width, f->height);
    }
  }
  if (vipc_server) {
    QThread *t = QThread::create(&Replay::cameraThread, this);
    connect(t, &QThread::finished, t, &QThread::deleteLater);
    t->start();
    qDebug() << "start vipc server";
  }
}

std::optional<std::pair<FrameReader *, uint32_t>> Replay::getFrame(int seg_id, FrameType type, uint32_t frame_id) {
  auto seg = getSegment(seg_id);
  if (!seg) return std::nullopt;

  const EncodeIdx *eidx = seg->log->getFrameEncodeIdx(type, frame_id);
  if (!eidx) return std::nullopt;

  auto frame_seg = (seg_id == (*eidx).segmentNum) ? seg : getSegment(eidx->segmentNum);
  if (!frame_seg) return std::nullopt;

  FrameReader *frm = frame_seg->frames[type];
  if (!frm) return std::nullopt;

  return std::make_pair(frm, eidx->segmentId);
};

void Replay::seekTime(int ts) {
  ts = std::clamp(ts, 0, log_paths.size() * SEGMENT_LENGTH);
  qInfo() << "seeking to " << ts;

  seek_ts = ts;
  current_segment = ts / SEGMENT_LENGTH;
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
    std::pair<FrameType, uint32_t> frame;
    if (!frame_queue.try_pop(frame, 50)) continue;

    // search frame's encodIdx in adjacent segments.
    auto [type, frame_id] = frame;
    int search_in[] = {current_segment, current_segment - 1, current_segment + 1};
    for (auto i : search_in) {
      if (i < 0 || i >= segments.size()) continue;

      if(auto f = getFrame(i, type, frame_id); f) {
        auto [frm, fid] = *f;
        VisionIpcBufExtra extra = {};
        VisionBuf *buf = vipc_server->get_buffer(frm->stream_type);
        memcpy(buf->addr, frm->get(fid), frm->getRGBSize());
        vipc_server->send(buf, &extra, false);
        break;
      }
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
    const SegmentData *seg = getSegment(current_segment);
    if (!seg) {
      qDebug() << "waiting for events";
      QThread::msleep(100);
      continue;
    }
    if (vipc_server == nullptr) {
      startVipcServer(seg);
    }
    const Events &events = seg->log->events();

    // TODO: use initData's logMonoTime
    if (route_start_ts == 0) {
      route_start_ts = events.firstKey();
    }

    uint64_t t0 = route_start_ts + (seek_ts * 1e9);
    qDebug() << "unlogging at" << (t0 - route_start_ts) / 1e9;

    auto eit = events.lowerBound(t0);
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

        // publish frames
        if (e.which() == cereal::Event::ROAD_CAMERA_STATE) {
          frame_queue.push({RoadCamFrame, e.getRoadCameraState().getFrameId()});
        } else if (e.which() == cereal::Event::DRIVER_CAMERA_STATE) {
          frame_queue.push({DriverCamFrame, e.getDriverCameraState().getFrameId()});
        } else if (e.which() == cereal::Event::WIDE_ROAD_CAMERA_STATE) {
          frame_queue.push({WideRoadCamFrame, e.getWideRoadCameraState().getFrameId()});
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
