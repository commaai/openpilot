#include "selfdrive/ui/replay/route.h"

#include <QEventLoop>
#include <QJsonArray>
#include <QJsonDocument>
#include <QRegExp>
#include <QtConcurrent>

#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/qt/api.h"
#include "selfdrive/ui/replay/util.h"

bool Route::load() {
  if (data_dir_.isEmpty()) {
    return loadFromServer();
  } else {
    return loadFromLocal();
  }
}

bool Route::loadFromServer() {
  QEventLoop loop;
  auto onError = [&loop](const QString &err) { loop.quit(); };

  bool ret = false;
  HttpRequest http(nullptr, !Hardware::PC());
  QObject::connect(&http, &HttpRequest::failedResponse, onError);
  QObject::connect(&http, &HttpRequest::timeoutResponse, onError);
  QObject::connect(&http, &HttpRequest::receivedResponse, [&](const QString json) {
    ret = loadFromJson(json);
    loop.quit();
  });
  http.sendRequest("https://api.commadotai.com/v1/route/" + route_ + "/files");
  loop.exec();
  return ret;
}

bool Route::loadFromJson(const QString &json) {
  QJsonObject route_files = QJsonDocument::fromJson(json.trimmed().toUtf8()).object();
  if (route_files.empty()) {
    qInfo() << "JSON Parse failed";
    return false;
  }

  QRegExp rx(R"(\/(\d+)\/)");
  for (const QString &key : route_files.keys()) {
    for (const auto &url : route_files[key].toArray()) {
      QString url_str = url.toString();
      if (rx.indexIn(url_str) != -1) {
        addFileToSegment(rx.cap(1).toInt(), url_str);
      }
    }
  }
  return true;
}

bool Route::loadFromLocal() {
  QString prefix = route_.split('|').last();
  if (prefix.isEmpty()) return false;

  QDir log_dir(data_dir_);
  QStringList folders = log_dir.entryList(QDir::Dirs | QDir::NoDot | QDir::NoDotDot, QDir::NoSort);
  if (folders.isEmpty()) return false;

  for (auto folder : folders) {
    int seg_num_pos = folder.lastIndexOf("--");
    if (seg_num_pos != -1) {
      const int seg_num = folder.mid(seg_num_pos + 2).toInt();
      QDir segment_dir(log_dir.filePath(folder));
      for (auto f : segment_dir.entryList(QDir::Files)) {
        addFileToSegment(seg_num, segment_dir.absoluteFilePath(f));
      }
    }
  }
  return true;
}

void Route::addFileToSegment(int n, const QString &file) {
  const QString name = QUrl(file).fileName();
  if (name == "rlog.bz2") {
    segments_[n].rlog = file;
  } else if (name == "qlog.bz2") {
    segments_[n].qlog = file;
  } else if (name == "fcamera.hevc") {
    segments_[n].road_cam = file;
  } else if (name == "dcamera.hevc") {
    segments_[n].driver_cam = file;
  } else if (name == "ecamera.hevc") {
    segments_[n].wide_road_cam = file;
  } else if (name == "qcamera.ts") {
    segments_[n].qcamera = file;
  }
}

// class Segment

Segment::Segment(int n, const SegmentFile &files, bool load_dcam, bool load_ecam) : seg_num(n) {
  static std::once_flag once_flag;
  std::call_once(once_flag, [=]() { if (!CACHE_DIR.exists()) QDir().mkdir(CACHE_DIR.absolutePath()); });

  // the order is [RoadCam, DriverCam, WideRoadCam, log]. fallback to qcamera/qlog
  const QString file_list[] = {
      files.road_cam.isEmpty() ? files.qcamera : files.road_cam,
      load_dcam ? files.driver_cam : "",
      load_ecam ? files.wide_road_cam : "",
      files.rlog.isEmpty() ? files.qlog : files.rlog,
  };
  for (int i = 0; i < std::size(file_list); i++) {
    if (!file_list[i].isEmpty()) {
      loading_++;
      synchronizer_.addFuture(QtConcurrent::run(this, &Segment::loadFile, i, file_list[i].toStdString()));
    }
  }
}

Segment::~Segment() {
  aborting_ = true;
  synchronizer_.setCancelOnWait(true);
  synchronizer_.waitForFinished();
}

void Segment::loadFile(int id, const std::string file) {
  const bool is_remote = file.find("https://") == 0;
  const std::string local_file = is_remote ? cacheFilePath(file) : file;
  bool file_ready = util::file_exists(local_file);

  if (!file_ready && is_remote) {
    // TODO: retry on failure
    file_ready = httpMultiPartDownload(file, local_file, id < MAX_CAMERAS ? 3 : 1, &aborting_);
  }

  if (!aborting_ && file_ready) {
    if (id < MAX_CAMERAS) {
      frames[id] = std::make_unique<FrameReader>();
      success_ = success_ && frames[id]->load(local_file);
    } else {
      std::string decompressed = cacheFilePath(local_file + ".decompressed");
      if (!util::file_exists(decompressed)) {
        std::ofstream ostrm(decompressed, std::ios::binary);
        readBZ2File(local_file, ostrm);
      }
      log = std::make_unique<LogReader>();
      success_ = success_ && log->load(decompressed);
    }
  }

  if (!aborting_ && --loading_ == 0) {
    emit loadFinished(success_);
  }
}

std::string Segment::cacheFilePath(const std::string &file) {
  QString url_no_query = QUrl(file.c_str()).toString(QUrl::RemoveQuery);
  QString sha256 = QCryptographicHash::hash(url_no_query.toUtf8(), QCryptographicHash::Sha256).toHex();
  return CACHE_DIR.filePath(sha256 + "." + QFileInfo(url_no_query).suffix()).toStdString();
}

std::shared_ptr<Segment> SegmentManager::get(int n, const SegmentFile &files, bool load_cam, bool load_eccam) {
  // get segment from cache
  if (auto it = segments_.find(n); it != segments_.end()) {
    qDebug() << "get segment" << n << "from cache";
    it->second->last_used = millis_since_boot();
    return it->second->segment;
  }

  // TODO:  dynamically adjust cache_size_ based on the amount of free memory
  // remove unused segments from cache
  while (segments_.size() >= cache_size_) {
    auto it = std::min_element(segments_.begin(), segments_.end(), [](auto &a, auto &smallest){
      return a.second->last_used < smallest.second->last_used;
    });
    if (it == segments_.end() || it->second->segment.use_count() > 1) break;
    // qDebug() << "remove segment" << it->first << "from cache";
    segments_.erase(it);
  }

  // add segment to cache
  qInfo() << "loading segment" << n << "...";
  SegmentData *s = new SegmentData;
  s->segment = std::make_shared<Segment>(n, files, load_cam, load_eccam),
  s->last_used = millis_since_boot(),
  QObject::connect(s->segment.get(), &Segment::loadFinished, [=](bool success) { emit segmentLoadFinished(n, success); });
  segments_.emplace(n, s);
  return s->segment;
}
