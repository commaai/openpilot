#include "selfdrive/ui/replay/route.h"

#include <QEventLoop>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QRegExp>
#include <QThread>
#include <future>

#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/qt/api.h"
#include "selfdrive/ui/replay/util.h"

Route::Route(const QString &route, const QString &data_dir) : route_(route), data_dir_(data_dir) {}

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
        const int seg_num = rx.cap(1).toInt();
        addFileToSegment(seg_num, url_str);
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
    const int seg_num = folder.split("--")[2].toInt();
    QDir segment_dir(log_dir.filePath(folder));
    for (auto f : segment_dir.entryList(QDir::Files)) {
      addFileToSegment(seg_num, segment_dir.absoluteFilePath(f));
    }
  }
  return true;
}

void Route::addFileToSegment(int seg_num, const QString &file) {
  if (segments_.size() <= seg_num) {
    segments_.resize(seg_num + 1);
  }

  QString fn = QUrl(file).fileName();
  if (fn == "rlog.bz2") {
    segments_[seg_num].rlog = file;
  } else if (fn == "qlog.bz2") {
    segments_[seg_num].qlog = file;
  } else if (fn == "fcamera.hevc") {
    segments_[seg_num].road_cam = file;
  } else if (fn == "dcamera.hevc") {
    segments_[seg_num].driver_cam = file;
  } else if (fn == "ecamera.hevc") {
    segments_[seg_num].wide_road_cam = file;
  } else if (fn == "qcamera.ts") {
    segments_[seg_num].qcamera = file;
  }
}

// class Segment

Segment::Segment(int n, const SegmentFile &files, bool load_dcam, bool load_ecam) : seg_num_(n) {
  static std::once_flag once_flag;
  std::call_once(once_flag, [=]() {
    if (!CACHE_DIR.exists()) QDir().mkdir(CACHE_DIR.absolutePath());
  });

  // fallback to qcamera/qlog
  const QString road_cam_path = files.road_cam.isEmpty() ? files.qcamera : files.road_cam;
  const QString log_path = files.rlog.isEmpty() ? files.qlog : files.rlog;
  assert(!log_path.isEmpty() && !road_cam_path.isEmpty());

  const QString driver_cam_path = load_dcam ? files.driver_cam : "";
  const QString wide_road_cam_path = load_ecam ? files.wide_road_cam : "";

  const QString file_list[] = {road_cam_path, driver_cam_path, wide_road_cam_path, log_path};
  for (int i = 0; i < std::size(file_list); ++i) {
    if (!file_list[i].isEmpty()) {
      ++loading_;
      threads_.emplace_back(QThread::create([=, fn = file_list[i]] { loadFile(i, fn.toStdString()); }))->start();
    }
  }
}

Segment::~Segment() {
  aborting_ = true;
  for (QThread *t : threads_) {
    if (t->isRunning()) t->wait();
    delete t;
  }
}

void Segment::loadFile(int id, const std::string file) {
  bool is_remote_file = file.find("https://") == 0;
  std::string local_file = is_remote_file ? cacheFilePath(file) : file;
  if (is_remote_file  && !util::file_exists(local_file)) {
    httpMultiPartDownload(file, local_file, connections_per_file, &aborting_);
  }

  if (!aborting_) {
    if (id == MAX_CAMERAS) {
      // pre-decompress log file.
      std::string decompressed_file = cacheFilePath(local_file + ".decompressed");
      if (!util::file_exists(decompressed_file)) {
        std::ofstream ostrm(decompressed_file, std::ios::binary);
        readBZ2File(local_file, ostrm);
      }
      log = std::make_unique<LogReader>();
      log->load(decompressed_file);
    } else {
      frames[id] = std::make_unique<FrameReader>();
      frames[id]->load(local_file);
    }
    if (--loading_ == 0 && !aborting_) {
      emit loadFinished();
    }
  }
}

std::string Segment::cacheFilePath(const std::string &file) {
  QUrl url(file.c_str());
  QByteArray url_no_query = url.toString(QUrl::RemoveQuery).toUtf8();
  QString sha256 = QCryptographicHash::hash(url_no_query, QCryptographicHash::Sha256).toHex();
  return CACHE_DIR.filePath(sha256 + "." + QFileInfo(url.fileName()).suffix()).toStdString();
}
