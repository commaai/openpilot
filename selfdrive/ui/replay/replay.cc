#include "selfdrive/ui/replay/replay.h"

#include <openssl/sha.h>
#include "curl/curl.h"

#include <QJsonDocument>
#include <QJsonObject>
#include <QtConcurrent>

#include "cereal/services.h"
#include "selfdrive/camerad/cameras/camera_common.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/hardware/hw.h"

const std::string CACHE_DIR = util::getenv("COMMA_CACHE", "/tmp/comma_download_cache/");

size_t write_data(void *ptr, size_t size, size_t nmemb, FILE *stream) {
  size_t written = fwrite(ptr, size, nmemb, stream);
  return written;
}

bool download_url(const std::string &url, const std::string &file) {
  CURLcode res = CURLE_FAILED_INIT;
  CURL *curl = curl_easy_init();
  if (curl) {
    FILE *fp = fopen(file.c_str(), "wb");
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    fclose(fp);
  }
  return CURLE_OK == res;
}

std::string sha256_string(const std::string &string) {
  uint8_t hash[SHA256_DIGEST_LENGTH] = {};
  SHA256_CTX sha256;
  SHA256_Init(&sha256);
  SHA256_Update(&sha256, string.c_str(), string.length());
  SHA256_Final(hash, &sha256);
  char outputBuffer[2 * SHA256_DIGEST_LENGTH + 1] = {};
  for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
    sprintf(outputBuffer + (i * 2), "%02x", hash[i]);
  }
  return outputBuffer;
}

std::optional<std::string> download_segment_file(const std::string &url, const std::string fn) {
  if (!util::file_exists(CACHE_DIR)) {
    system(("mkdir " + CACHE_DIR).c_str());
  }
  std::string file = CACHE_DIR + sha256_string(fn);
  if (util::file_exists(file)) return file;

  std::string tmp_file = file + ".tmp";
  if (util::file_exists(tmp_file)) unlink(tmp_file.c_str());
  bool ret = download_url(url, tmp_file);
  ret = ret && (rename(tmp_file.c_str(), file.c_str()) == 0);
  return ret ? std::make_optional(file) : std::nullopt;
}

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

Replay::Replay(QString route, SubMaster *sm_, QObject *parent) : sm(sm_), route_(route), QObject(parent) {
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

  const QString url = CommaApi::BASE_URL + "/v1/route/" + route + "/files";
  http = new HttpRequest(this, !Hardware::PC());
  QObject::connect(http, &HttpRequest::receivedResponse, this, &Replay::parseResponse);
  http->sendRequest(url);
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
  {
    std::lock_guard lk(merge_mutex);
    assert((n >= 0) && (n < log_paths.size()) && (n < camera_paths.size()));
    if (lrs.find(n) != lrs.end()) return;

    lrs[n] = new LogReader();
    frs[n] = new FrameReader();
  }

  std::string segment_path = util::string_format("%s--%d", route_.toStdString().c_str(), n);
  std::string log_url = log_paths.at(n).toString().toStdString();
  std::string road_cam_url = camera_paths.at(n).toString().toStdString();

  std::optional<std::string> log_file = download_segment_file(log_url, segment_path + "/rlog.zip");
  std::optional<std::string> road_cam_file = download_segment_file(road_cam_url, segment_path + "/road_cam.hevc");
  if (log_file && road_cam_file && lrs[n]->load(log_file->c_str()) && frs[n]->load(road_cam_file->c_str())) {
    mergeEvents();
  } else {
    LOGW("failed to load segment: %s", segment_path.c_str());
  }
}

void Replay::mergeEvents() {
  const int start_idx = std::max(current_segment - BACKWARD_SEGS, 0);
  const int end_idx = std::min(current_segment + FORWARD_SEGS, log_paths.size());

  // merge logs
  QMultiMap<uint64_t, Event *> *new_events = new QMultiMap<uint64_t, Event *>();
  std::unordered_map<uint32_t, EncodeIdx> *new_eidx = new std::unordered_map<uint32_t, EncodeIdx>[MAX_CAMERAS];
  for (int i = start_idx; i <= end_idx; ++i) {
    if (auto it = lrs.find(i); it != lrs.end()) {
      *new_events += (*it)->events;
      for (CameraType cam_type : ALL_CAMERAS) {
        new_eidx[cam_type].merge((*it)->eidx[cam_type]);
      }
    }
  }

  // update logs
  updating_events = true; // set updating_events to true to force stream thread relase the lock
  lock.lock();
  auto prev_events = std::exchange(events, new_events);
  auto prev_eidx = std::exchange(eidx, new_eidx);
  lock.unlock();

  // free logs
  delete prev_events;
  delete[] prev_eidx;
  for (int i = 0; i < log_paths.size(); i++) {
    if (i < start_idx || i > end_idx) {
      delete lrs.take(i);
      delete frs.take(i);
    }
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
  updating_events = true;
}

void Replay::segmentQueueThread() {
  // maintain the segment window
  while (true) {
    int start_idx = std::max(current_segment - BACKWARD_SEGS, 0);
    int end_idx = std::min(current_segment + FORWARD_SEGS, log_paths.size());
    for (int i = 0; i < log_paths.size(); i++) {
      if (i >= start_idx && i <= end_idx) {
        QtConcurrent::run(this, &Replay::addSegment, i);
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
  uint64_t cur_mono_time = 0;
  while (true) {
    std::unique_lock lk(lock);

    if (!events || events->size() == 0) {
      qDebug() << "waiting for events";
      QThread::msleep(100);
      continue;
    }

    // TODO: use initData's logMonoTime
    if (route_start_ts == 0) {
      route_start_ts = events->firstKey();
    }

    uint64_t t0 = seek_ts != -1 ? route_start_ts + (seek_ts * 1e9) : cur_mono_time;
    seek_ts = -1;
    qDebug() << "unlogging at" << int((t0 - route_start_ts) / 1e9);
    uint64_t t0r = timer.nsecsElapsed();

    for (auto eit = events->lowerBound(t0); !updating_events && eit != events->end(); ++eit) {
      cereal::Event::Reader e = (*eit)->event;
      cur_mono_time = (*eit)->mono_time;
      std::string type;
      KJ_IF_MAYBE(e_, static_cast<capnp::DynamicStruct::Reader>(e).which()) {
        type = e_->getProto().getName();
      }

      current_ts = std::max(cur_mono_time - route_start_ts, (uint64_t)0) / 1e9;

      if (socks.contains(type)) {
        float timestamp = (cur_mono_time - route_start_ts)/1e9;
        if (std::abs(timestamp - last_print) > 5.0) {
          last_print = timestamp;
          qInfo() << "at " << int(last_print) << "s";
        }

        // keep time
        long etime = cur_mono_time-t0;
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
    }
    updating_events = false;
    usleep(0);
  }
}
