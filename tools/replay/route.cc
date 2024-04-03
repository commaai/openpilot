#include "tools/replay/route.h"

#include <QDir>
#include <QEventLoop>
#include <QJsonArray>
#include <QJsonDocument>
#include <QRegularExpression>
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
  RouteIdentifier identifier = {};
  QRegularExpression rx(R"(^((?<dongle_id>[a-z0-9]{16})[|_/])?(?<timestamp>.{20})((?<separator>--|/)(?<range>((-?\d+(:(-?\d+)?)?)|(:-?\d+))))?$)");
  if (auto match = rx.match(str); match.hasMatch()) {
    identifier.dongle_id = match.captured("dongle_id");
    identifier.timestamp = match.captured("timestamp");
    identifier.str = identifier.dongle_id + "|" + identifier.timestamp;
    auto range_str = match.captured("range");
    if (auto separator = match.captured("separator"); separator == "/" && !range_str.isEmpty()) {
      auto range = range_str.split(":");
      identifier.begin_segment = identifier.end_segment = range[0].toInt();
      if (range.size() == 2) {
        identifier.end_segment = range[1].isEmpty() ? -1 : range[1].toInt();
      }
    } else if (separator == "--") {
      identifier.begin_segment = range_str.toInt();
    }
  }
  return identifier;
}

bool Route::load() {
  if (route_.str.isEmpty() || (data_dir_.isEmpty() && route_.dongle_id.isEmpty())) {
    rInfo("invalid route format");
    return false;
  }
  date_time_ = QDateTime::fromString(route_.timestamp, "yyyy-MM-dd--HH-mm-ss");
  bool ret = data_dir_.isEmpty() ? loadFromServer() : loadFromLocal();
  if (ret) {
    if (route_.begin_segment == -1) route_.begin_segment = segments_.rbegin()->first;
    if (route_.end_segment == -1) route_.end_segment = segments_.rbegin()->first;
    for (auto it = segments_.begin(); it != segments_.end(); /**/) {
      if (it->first < route_.begin_segment || it->first > route_.end_segment) {
        it = segments_.erase(it);
      } else {
        ++it;
      }
    }
  }
  return !segments_.empty();
}

bool Route::loadFromServer(int retries) {
  for (int i = 1; i <= retries; ++i) {
    QString result;
    QEventLoop loop;
    HttpRequest http(nullptr, !Hardware::PC());
    QObject::connect(&http, &HttpRequest::requestDone, [&loop, &result](const QString &json, bool success, QNetworkReply::NetworkError err) {
      result = json;
      loop.exit((int)err);
    });
    http.sendRequest(CommaApi::BASE_URL + "/v1/route/" + route_.str + "/files");
    auto err = (QNetworkReply::NetworkError)loop.exec();
    if (err == QNetworkReply::NoError) {
      return loadFromJson(result);
    } else if (err == QNetworkReply::ContentAccessDenied || err == QNetworkReply::AuthenticationRequiredError) {
      rWarning(">>  Unauthorized. Authenticate with tools/lib/auth.py  <<");
      return false;
    }
    rWarning("Retrying %d/%d", i, retries);
    util::sleep_for(500);
  }
  return false;
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
  QDirIterator it(data_dir_, {QString("%1--*").arg(route_.timestamp)}, QDir::Dirs | QDir::NoDotAndDotDot);
  while (it.hasNext()) {
    QString segment = it.next();
    const int seg_num = segment.mid(segment.lastIndexOf("--") + 2).toInt();
    QDir segment_dir(segment);
    for (const auto &f : segment_dir.entryList(QDir::Files)) {
      addFileToSegment(seg_num, segment_dir.absoluteFilePath(f));
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
