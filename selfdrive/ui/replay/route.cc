#include "selfdrive/ui/replay/route.h"

#include <QDir>
#include <QEventLoop>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QRegExp>
#include <future>

#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/qt/api.h"

Route::Route(const QString &route) : route_(route) {}

bool Route::load() {
  QEventLoop loop;
  auto onError = [&loop](const QString &err) {
    qInfo() << err;
    loop.quit();
  };

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
        if (segments_.size() <= seg_num) {
          segments_.resize(seg_num + 1);
        }
        if (key == "logs") {
          segments_[seg_num].rlog = url_str;
        } else if (key == "qlogs") {
          segments_[seg_num].qlog = url_str;
        } else if (key == "cameras") {
          segments_[seg_num].road_cam = url_str;
        } else if (key == "dcameras") {
          segments_[seg_num].driver_cam = url_str;
        } else if (key == "ecameras") {
          segments_[seg_num].wide_road_cam = url_str;
        } else if (key == "qcameras") {
          segments_[seg_num].qcamera = url_str;
        }
      }
    }
  }
  return true;
}

// class Segment

Segment::Segment(int n, const SegmentFile &segment_files) : seg_num_(n), files_(segment_files) {
  static std::once_flag once_flag;
  std::call_once(once_flag, [=]() {
    if (!QDir(CACHE_DIR).exists()) QDir().mkdir(CACHE_DIR);
  });

  // fallback to qcamera
  road_cam_path_ = files_.road_cam.isEmpty() ? files_.qcamera : files_.road_cam;
  valid_ = !files_.rlog.isEmpty() && !road_cam_path_.isEmpty();
  if (!valid_) return;

  if (!QUrl(files_.rlog).isLocalFile()) {
    for (auto &url : {files_.rlog, road_cam_path_, files_.driver_cam, files_.wide_road_cam}) {
      if (!url.isEmpty() && !QFile::exists(localPath(url))) {
        qDebug() << "download" << url;
        downloadFile(url);
        ++downloading_;
      }
    }
  }
  if (downloading_ == 0) {
    QTimer::singleShot(0, this, &Segment::load);
  }
}

Segment::~Segment() {
  // cancel download, qnam will not abort requests, need to abort them manually
  aborting_ = true;
  for (QNetworkReply *replay : replies_) {
    if (replay->isRunning()) {
      replay->abort();
    }
    replay->deleteLater();
  }
}

void Segment::downloadFile(const QString &url) {
  QNetworkReply *reply = qnam_.get(QNetworkRequest(url));
  replies_.insert(reply);
  connect(reply, &QNetworkReply::finished, [=]() {
    if (reply->error() == QNetworkReply::NoError) {
      QFile file(localPath(url));
      if (file.open(QIODevice::WriteOnly)) {
        file.write(reply->readAll());
      }
    }
    if (--downloading_ == 0 && !aborting_) {
      load();
    }
  });
}

// load concurrency
void Segment::load() {
  std::vector<std::future<bool>> futures;
  futures.emplace_back(std::async(std::launch::async, [=]() {
    log = std::make_unique<LogReader>();
    return log->load(localPath(files_.rlog).toStdString());
  }));

  QString camera_files[] = {road_cam_path_, files_.driver_cam, files_.wide_road_cam};
  for (int i = 0; i < std::size(camera_files); ++i) {
    if (!camera_files[i].isEmpty()) {
      futures.emplace_back(std::async(std::launch::async, [=]() {
        frames[i] = std::make_unique<FrameReader>();
        return frames[i]->load(localPath(camera_files[i]).toStdString());
      }));
    }
  }

  int success_cnt = std::accumulate(futures.begin(), futures.end(), 0, [=](int v, auto &f) { return f.get() + v; });
  loaded_ = valid_ = (success_cnt == futures.size());
  emit loadFinished();
}

QString Segment::localPath(const QUrl &url) {
  if (url.isLocalFile()) return url.toString();

  QByteArray url_no_query = url.toString(QUrl::RemoveQuery).toUtf8();
  return CACHE_DIR + QString(QCryptographicHash::hash(url_no_query, QCryptographicHash::Sha256).toHex());
}
