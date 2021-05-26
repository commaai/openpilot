#include "selfdrive/ui/replay/replay.h"

#include <QJsonDocument>
#include <QJsonObject>
#include <QDir>

#include "cereal/services.h"
#include "selfdrive/camerad/cameras/camera_common.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/hardware/hw.h"

const int SEGMENT_LENGTH = 60; // 60s
const int FORWARD_SEGS = 2;
const int BACKWARD_SEGS = 2;
const std::string LOG_ROOT =
    Hardware::PC() ? util::getenv_default("HOME", "/.comma/media/0/realdata", "/data/media/0/realdata")
                   : "/data/media/0/realdata";

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
  delete pm;
  CL_CHECK(clReleaseContext(context));
}

void Replay::load() {
  if (!loadFromLocal()) {
    loadFromServer();
  }
}

void Replay::loadFromServer() {
  const QString url = "https://api.commadotai.com/v1/route/" + route + "/files";
  http = new HttpRequest(this, url, "", !Hardware::PC());
  QObject::connect(http, &HttpRequest::receivedResponse, this, &Replay::loadFromJson);
}

bool Replay::loadFromLocal() {
  QStringList list = route.split('|');
  if (list.size() != 2) return false;

  QJsonArray cameras, dcameras, ecameras, qcameras;
  QJsonArray logs, qlogs;

  QDir log_root(LOG_ROOT.c_str());
  const  QStringList folders = log_root.entryList(QStringList() << list[1] + "*", QDir::Dirs | QDir::NoDot);

  for (auto folder : folders) {
    QDir segment(log_root.filePath(folder));
    const QStringList files = segment.entryList(QDir::Files);
    for (auto f : files) {
      const QString file_path = "file://" + segment.filePath(f);
      if (f.startsWith("fcamera")) {
        cameras << file_path;
      } else if (f.startsWith("dcamera")) {
        dcameras << file_path;
      } else if (f.startsWith("ecamera")) {
        ecameras << file_path;
      } else if (f.startsWith("qcamera")) {
        qcameras << file_path;
      } else if (f.startsWith("rlog")) {
        logs << file_path;
      } else if (f.startsWith("qlog")) {
        qlogs << file_path;
      }
    }
  }

  QJsonObject obj;
  obj["cameras"] = cameras;
  obj["dcameras"] = dcameras;
  obj["ecameras"] = ecameras;
  obj["qcameras"] = qcameras;
  obj["logs"] = logs;
  obj["qlogs"] = qlogs;

  QJsonDocument doc(obj);
  QString json = doc.toJson(QJsonDocument::Compact);
  return loadFromJson(json);
}

bool Replay::loadFromJson(const QString &json) {
  QJsonDocument doc = QJsonDocument::fromJson(json.trimmed().toUtf8());
  if (doc.isNull()) {
    qInfo() << "JSON Parse failed";
    return false;
  }

  frame_paths[RoadCamFrame] = doc["cameras"].toVariant().toStringList();
  if (frame_paths[RoadCamFrame].isEmpty()) {
    // fallback to qcameras
    frame_paths[RoadCamFrame] = doc["qcameras"].toVariant().toStringList();
  }
  frame_paths[DriverCamFrame] = doc["dcameras"].toVariant().toStringList();
  frame_paths[WideRoadCamFrame] = doc["ecameras"].toVariant().toStringList();
  
  log_paths = doc["logs"].toVariant().toStringList();
  if (log_paths.isEmpty()) {
    // fallback to qlogs
    log_paths = doc["qlogs"].toVariant().toStringList();
  }

  if (log_paths.isEmpty()) {
    qInfo() << "no logs found in route " << route;
    return false;
  }
  
  qInfo() << "replay route " << route << ", total segments:" << log_paths.size();

  // start threads
  typedef void (Replay::*threadFunc)();
  threadFunc thread_func[] = {&Replay::segmentQueueThread, &Replay::keyboardThread, &Replay::streamThread};
  for (int i = 0; i < std::size(thread_func); ++i) {
    QThread *t = QThread::create(thread_func[i], this);
    connect(t, &QThread::finished, t, &QThread::deleteLater);
    t->start();
  }
  return true;
}

void Replay::addSegment(int n) {
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
  connect(seg->log, &LogReader::done, [=] { 
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
    connect(reader, &FrameReader::done, [=] { --seg->loading; });
    QThread *t = QThread::create([=] { reader->process(); });
    QObject::connect(t, &QThread::finished, t, &QThread::deleteLater);
    t->start();
    return reader;
  };
  for (int i = 0; i < std::size(frame_paths); ++i) {
    if (n < frame_paths[i].size()) {
      seg->frames[i] = read_frames(frame_paths[i][n], VISION_STREAM_RGB_BACK);  
    }  
  }
}

// return nullptr if segment is not loaded
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
  while (!exit_) {
    for (int i = 0; i < log_paths.size(); i++) {
      int start_idx = std::max(current_segment - BACKWARD_SEGS, 0);
      int end_idx = std::min(current_segment + FORWARD_SEGS, log_paths.size());
      if (i >= start_idx && i <= end_idx) {
        addSegment(i);
      } else if (i != playing_segment) {
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

    auto [frame_type, frame_id] = frame;
    // search frame's encodIdx in adjacent segments.
    int search_in[] = {current_segment, current_segment - 1, current_segment + 1};
    for (auto i : search_in) {
      if (i < 0 || i >= segments.size()) continue;

      if(auto f = getFrame(i, frame_type, frame_id)) {
        auto [frame_reader, fid] = *f;
        if (uint8_t *data = frame_reader->get(fid)) {
          VisionIpcBufExtra extra = {};
          VisionBuf *buf = vipc_server->get_buffer(frame_reader->stream_type);
          memcpy(buf->addr, frame_reader->get(fid), frame_reader->getRGBSize());
          vipc_server->send(buf, &extra, false);
        } else {
          qDebug() << "failed to get frame " << frame_id << " from segment " << i;
        }
        break;
      }
    }
  }

  delete vipc_server;
  vipc_server = nullptr;
}

void Replay::keyboardThread() {
  char c;
  while (!exit_) {
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
  while (!exit_) {
    playing_segment = current_segment.load();
    const SegmentData *seg = getSegment(playing_segment);
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
    while (!exit_ && current_seek_ts == seek_ts && eit != events.end()) {
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
          qInfo() << "at " << last_print << "| segment:" << playing_segment;
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

    if (eit == events.end()) {
      // move to the next segment
      current_segment += 1;
      qDebug() << "move to next segment " << current_segment;
      seek_ts = current_ts.load();
    } else {
    }
  }
}
