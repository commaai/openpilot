#include "selfdrive/ui/replay/route.h"

#include <QEventLoop>
#include <QJsonArray>
#include <QJsonDocument>
#include <QRegExp>

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
  for (const auto &value :  QJsonDocument::fromJson(json.trimmed().toUtf8()).object()) {
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

Segment::Segment(int n, const SegmentFile &files, bool load_dcam, bool load_ecam) : seg_num(n) {
  static std::once_flag once_flag;
  std::call_once(once_flag, [=]() { if (!CACHE_DIR.exists()) QDir().mkdir(CACHE_DIR.absolutePath()); });

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
      loading_threads_.emplace_back(QThread::create(&Segment::loadFile, this, i, file_list[i].toStdString()))->start();
    }
  }
}

Segment::~Segment() {
  aborting_ = true;
  for (QThread *t : loading_threads_) {
    if (t->isRunning()) t->wait();
    delete t;
  }
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
