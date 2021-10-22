#include "selfdrive/ui/replay/route.h"

#include <QEventLoop>
#include <QJsonArray>
#include <QJsonDocument>
#include <QRegExp>
#include <fstream>
#include <sstream>

#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/qt/api.h"
#include "selfdrive/ui/replay/util.h"

Route::Route(const QString &route, const QString &data_dir) : route_(parseRoute(route)), data_dir_(data_dir) {}

RouteIdentifier Route::parseRoute(const QString &str) {
  QRegExp rx(R"(^([a-z0-9]{16})([|_/])(\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2})(?:(--|/)(\d*))?$)");
  if (rx.indexIn(str) == -1) return {};

  const QStringList list = rx.capturedTexts();
  return {list[1], list[3], list[5].toInt(), list[1] + "|" + list[3]};
}

bool Route::load() {
  if (route_.str.isEmpty()) {
    qInfo() << "invalid route format";
    return false;
  }
  return data_dir_.isEmpty() ? loadFromServer() : loadFromLocal();
}

bool Route::loadFromServer() {
  QEventLoop loop;
  HttpRequest http(nullptr, !Hardware::PC());
  QObject::connect(&http, &HttpRequest::failedResponse, [&] { loop.exit(0); });
  QObject::connect(&http, &HttpRequest::timeoutResponse, [&] { loop.exit(0); });
  QObject::connect(&http, &HttpRequest::receivedResponse, [&](const QString &json) {
    loop.exit(loadFromJson(json));
  });
  http.sendRequest("https://api.commadotai.com/v1/route/" + route_.str + "/files");
  return loop.exec();
}

bool Route::loadFromJson(const QString &json) {
  QRegExp rx(R"(\/(\d+)\/)");
  for (const auto &value : QJsonDocument::fromJson(json.trimmed().toUtf8()).object()) {
    for (const auto &url : value.toArray()) {
      QString url_str = url.toString();
      if (rx.indexIn(url_str) != -1) {
        addFileToSegment(rx.cap(1).toInt(), url_str);
      }
    }
  }
  return !segments_.empty();
}

bool Route::loadFromLocal() {
  QDir log_dir(data_dir_);
  for (const auto &folder : log_dir.entryList(QDir::Dirs | QDir::NoDot | QDir::NoDotDot, QDir::NoSort)) {
    int pos = folder.lastIndexOf("--");
    if (pos != -1 && folder.left(pos) == route_.timestamp) {
      const int seg_num = folder.mid(pos + 2).toInt();
      QDir segment_dir(log_dir.filePath(folder));
      for (const auto &f : segment_dir.entryList(QDir::Files)) {
        addFileToSegment(seg_num, segment_dir.absoluteFilePath(f));
      }
    }
  }
  return !segments_.empty();
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

Segment::Segment(int n, const SegmentFile &files, bool load_dcam, bool load_ecam, bool no_cache) : seg_num(n), no_local_cache_(no_cache) {
  static std::once_flag once_flag;
  std::call_once(once_flag, [=]() { if (!no_local_cache_ && !CACHE_DIR.exists()) QDir().mkdir(CACHE_DIR.absolutePath()); });

  // [RoadCam, DriverCam, WideRoadCam, log]. fallback to qcamera/qlog
  const QString file_list[] = {
      files.road_cam.isEmpty() ? files.qcamera : files.road_cam,
      load_dcam ? files.driver_cam : "",
      load_ecam ? files.wide_road_cam : "",
      files.rlog.isEmpty() ? files.qlog : files.rlog,
  };
  for (int i = 0; i < std::size(file_list); i++) {
    if (!file_list[i].isEmpty()) {
      loading_++;
      QThread *t = new QThread();
      QObject::connect(t, &QThread::started, [=]() { loadFile(i, file_list[i].toStdString()); });
      loading_threads_.emplace_back(t)->start();
    }
  }
}

Segment::~Segment() {
  aborting_ = true;
  for (QThread *t : loading_threads_) {
    if (t->isRunning()) {
      t->quit();
      t->wait();
    }
    delete t;
  }
}

void Segment::loadFile(int id, const std::string file) {
  const bool is_cam_file = id < MAX_CAMERAS;
  const bool is_remote = file.find("https://") == 0;
  const std::string local_file = is_remote ? cacheFilePath(file) : file;
  std::string file_content;
  
  if ((!is_remote || !no_local_cache_) && util::file_exists(local_file)) {
    file_content = util::read_file(local_file);
  } else if (is_remote) {
    bool decompress = !is_cam_file;
    file_content = downloadFile(file, no_local_cache_ ? "" : local_file, decompress);
  }

  if (!aborting_ && !file_content.empty()) {
    if (is_cam_file) {
      frames[id] = std::make_unique<FrameReader>();
      success_ = success_ && frames[id]->loadFromBuffer(std::move(file_content));
    } else {
      log = std::make_unique<LogReader>();
      success_ = success_ && log->load(std::move(file_content));
    }
  }

  if (!aborting_ && --loading_ == 0) {
    emit loadFinished(success_);
  }
}

std::string Segment::downloadFile(const std::string &url, const std::string &local_file, bool decompress) {
  const int chunk_size = 20 * 1024 * 1024; // 20MB
  std::string content;
  size_t remote_file_size = 0;

  for (int i = 1; i <= max_retries_; ++i) {
    if (remote_file_size <= 0) {
      remote_file_size = getRemoteFileSize(url);
    } 
    if (remote_file_size > 0 && !aborting_) {
      std::ostringstream oss;
      content.resize(remote_file_size);
      oss.rdbuf()->pubsetbuf(content.data(), content.size());
      int chunks = std::nearbyint(remote_file_size / (float)chunk_size);
      bool ret = httpMultiPartDownload(url, oss, chunks, remote_file_size, &aborting_);
      if (ret) {
        if (decompress) {
          content = decompressBZ2(content);
        }
        if (!local_file.empty()) {
          std::ofstream fs(local_file, fs.binary | fs.out);
          fs.write(content.data(), content.size());
        }
        return content;
      }
    }
    if (aborting_) break;

    qInfo() << "download failed, retrying" << i;
  }
  return {};
}

std::string Segment::cacheFilePath(const std::string &file) {
  QString url_no_query = QUrl(file.c_str()).toString(QUrl::RemoveQuery);
  QString sha256 = QCryptographicHash::hash(url_no_query.toUtf8(), QCryptographicHash::Sha256).toHex();
  return CACHE_DIR.filePath(sha256 + "." + QFileInfo(url_no_query).suffix()).toStdString();
}
