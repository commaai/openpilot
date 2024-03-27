#include "tools/replay/route.h"

#include <QDir>
#include <QEventLoop>
#include <QJsonArray>
#include <QJsonDocument>
#include <QRegExp>
#include <QtConcurrent>
#include <array>

#include "selfdrive/ui/qt/api.h"
#include "system/hardware/hw.h"
#include "tools/replay/replay.h"
#include "tools/replay/util.h"

Route::Route(const QString &route, const QString &data_dir) : data_dir_(data_dir) {
  route_ = parseRoute(route);
}

RouteIdentifier Route::parseRoute(const QString &str) {
  QRegExp rx(R"(^(?:([a-z0-9]{16})([|_/]))?(.{20})(?:(--|/)(\d*))?$)");
  if (rx.indexIn(str) == -1) return {};

  const QStringList list = rx.capturedTexts();
  return {.dongle_id = list[1], .timestamp = list[3], .segment_id = list[5].toInt(), .str = list[1] + "|" + list[3]};
}

bool Route::load() {
  if (route_.str.isEmpty() || (data_dir_.isEmpty() && route_.dongle_id.isEmpty())) {
    rInfo("invalid route format");
    return false;
  }
  date_time_ = QDateTime::fromString(route_.timestamp, "yyyy-MM-dd--HH-mm-ss");
  return data_dir_.isEmpty() ? loadFromServer() : loadFromLocal();
}

bool Route::loadFromServer() {
  QEventLoop loop;
  HttpRequest http(nullptr, !Hardware::PC());
  QObject::connect(&http, &HttpRequest::requestDone, [&](const QString &json, bool success, QNetworkReply::NetworkError error) {
    if (error == QNetworkReply::ContentAccessDenied || error == QNetworkReply::AuthenticationRequiredError) {
      qWarning() << ">>  Unauthorized. Authenticate with tools/lib/auth.py  <<";
    }

    loop.exit(success ? loadFromJson(json) : 0);
  });
  http.sendRequest(CommaApi::BASE_URL + "/v1/route/" + route_.str + "/files");
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
  QString name = QUrl(file).fileName();

  const int pos = name.lastIndexOf("--");
  name = pos != -1 ? name.mid(pos + 2) : name;

  if (name == "rlog.bz2" || name == "rlog") {
    segments_[n].rlog = file;
  } else if (name == "qlog.bz2" || name == "qlog") {
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

Segment::Segment(int n, const SegmentFile &files, uint32_t flags) : seg_num(n), flags(flags) {
  // [RoadCam, DriverCam, WideRoadCam, log]. fallback to qcamera/qlog
  const std::array file_list = {
      (flags & REPLAY_FLAG_QCAMERA) || files.road_cam.isEmpty() ? files.qcamera : files.road_cam,
      flags & REPLAY_FLAG_DCAM ? files.driver_cam : "",
      flags & REPLAY_FLAG_ECAM ? files.wide_road_cam : "",
      files.rlog.isEmpty() ? files.qlog : files.rlog,
  };
  for (int i = 0; i < file_list.size(); ++i) {
    if (!file_list[i].isEmpty() && (!(flags & REPLAY_FLAG_NO_VIPC) || i >= MAX_CAMERAS)) {
      ++loading_;
      synchronizer_.addFuture(QtConcurrent::run(this, &Segment::loadFile, i, file_list[i].toStdString()));
    }
  }
}

Segment::~Segment() {
  disconnect();
  abort_ = true;
  synchronizer_.setCancelOnWait(true);
  synchronizer_.waitForFinished();
}

void Segment::loadFile(int id, const std::string file) {
  const bool local_cache = !(flags & REPLAY_FLAG_NO_FILE_CACHE);
  bool success = false;
  if (id < MAX_CAMERAS) {
    frames[id] = std::make_unique<FrameReader>();
    success = frames[id]->load(file, flags & REPLAY_FLAG_NO_HW_DECODER, &abort_, local_cache, 20 * 1024 * 1024, 3);
  } else {
    log = std::make_unique<LogReader>();
    success = log->load(file, &abort_, local_cache, 0, 3);
  }

  if (!success) {
    // abort all loading jobs.
    abort_ = true;
  }

  if (--loading_ == 0) {
    emit loadFinished(!abort_);
  }
}
