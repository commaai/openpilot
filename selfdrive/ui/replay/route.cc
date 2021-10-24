#include "selfdrive/ui/replay/route.h"

#include <QDir>
#include <QEventLoop>
#include <QJsonArray>
#include <QJsonDocument>
#include <QRegExp>
#include <QtConcurrent>

#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/qt/api.h"
#include "selfdrive/ui/replay/util.h"

Route::Route(const QString &route, const QString &data_dir) : data_dir_(data_dir) {
  route_ = parseRoute(route);
}

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

Segment::Segment(int n, const SegmentFile &files, bool load_dcam, bool load_ecam, bool no_file_cache) : seg_num(n) {
  thread_pool_.setMaxThreadCount(2 + load_dcam + load_ecam);
  synchronizer_.setCancelOnWait(true);

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
      synchronizer_.addFuture(QtConcurrent::run(&thread_pool_, [=] { loadFile(i, file_list[i].toStdString(), no_file_cache); }));
    }
  }
}

Segment::~Segment() {
  disconnect();
  abort_ = true;
  synchronizer_.waitForFinished();
}

void Segment::loadFile(int id, const std::string file, bool no_file_cache) {
  std::call_once(once_flag_, [=] { qInfo() << "loading segment" << seg_num << "..."; });

  bool success = false;
  if (id < MAX_CAMERAS) {
    frames[id] = std::make_unique<FrameReader>(!no_file_cache, 20 * 1024 * 1024, 3);
    success = frames[id]->load(file, &abort_);
  } else {
    log = std::make_unique<LogReader>(!no_file_cache, -1, 3);
    success = log->load(file, &abort_);
  }

  if (!success) {
    // abort all loading jobs.
    abort_ = true;
  } 

  if (--loading_ == 0) {
    emit loadFinished(!abort_);
  }
}
