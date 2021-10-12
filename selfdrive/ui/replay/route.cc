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

Route::Route(const QString &route, const QString &data_dir) : route_(route), data_dir_("/home/batman/openpilot/xx/projects/ctf/out") {}

bool Route::load() {
  if (data_dir_.isEmpty()) {
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
  } else {
    return loadFromLocal();
  }
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

bool Route::loadFromLocal() {
  QString prefix = route_.split('|').last();
  if (prefix.isEmpty()) return false;

  QDir log_dir(data_dir_);
  QStringList folders = log_dir.entryList(QDir::Dirs | QDir::NoDot | QDir::NoDotDot, QDir::NoSort);
  if (folders.isEmpty()) return false;

  for (auto folder : folders) {
    const int seg_num = folder.split("--")[2].toInt();
    if (segments_.size() <= seg_num) {
      segments_.resize(seg_num + 1);
    }
    QDir segment_dir(log_dir.filePath(folder));
    for (auto f : segment_dir.entryList(QDir::Files)) {
      const QString file_path = segment_dir.absoluteFilePath(f);
      if (f.startsWith("rlog")) {
        segments_[seg_num].rlog = file_path;
      } else if (f.startsWith("qlog")) {
        segments_[seg_num].qlog = file_path;
      } else if (f.startsWith("fcamera")) {
        segments_[seg_num].road_cam = file_path;
      } else if (f.startsWith("dcamera")) {
        segments_[seg_num].driver_cam = file_path;
      } else if (f.startsWith("ecamera")) {
        segments_[seg_num].wide_road_cam = file_path;
      } else if (f.startsWith("qcamera")) {
        segments_[seg_num].qcamera = file_path;
      }
    }
  }
  return true;
}

// class Segment

Segment::Segment(int n, const SegmentFile &segment_files, bool load_dcam, bool load_ecam) : seg_num_(n), files_(segment_files) {
  static std::once_flag once_flag;
  std::call_once(once_flag, [=]() {
    if (!CACHE_DIR.exists()) QDir().mkdir(CACHE_DIR.absolutePath());
  });

  // fallback to qcamera/qlog
  road_cam_path_ = files_.road_cam.isEmpty() ? files_.qcamera : files_.road_cam;
  log_path_ = files_.rlog.isEmpty() ? files_.qlog : files_.rlog;
  assert (!log_path_.isEmpty() && !road_cam_path_.isEmpty());

  if (!load_dcam) {
    files_.driver_cam = "";
  }
  if (!load_ecam) {
    files_.wide_road_cam = "";
  }

  if (!QUrl(log_path_).isLocalFile()) {
    for (auto &url : {log_path_, road_cam_path_, files_.driver_cam, files_.wide_road_cam}) {
      if (!url.isEmpty() && !QFile::exists(localPath(url))) {
        downloadFile(url);
        ++downloading_;
      }
    }
  }
  if (downloading_ == 0) {
    QTimer::singleShot(0, this, &Segment::load);
  } else {
    qDebug() << "downloading segment" << seg_num_ << "...";
  }
}

Segment::~Segment() {
  aborting_ = true;
  if (downloading_ > 0) {
    qDebug() << "cancel download segment" << seg_num_;
  }
  for (auto &t : download_threads_) {
    if (t->isRunning()) t->wait();
  }
}

void Segment::downloadFile(const QString &url) {
  download_threads_.emplace_back(QThread::create([=]() {
    const std::string local_file = localPath(url).toStdString();
    bool ret = httpMultiPartDownload(url.toStdString(), local_file, connections_per_file, &aborting_);
    if (ret && url == log_path_) {
      // pre-decompress log file.
      std::ofstream ostrm(local_file + "_decompressed", std::ios::binary);
      readBZ2File(local_file, ostrm);
    }
    if (--downloading_ == 0 && !aborting_) {
      load();
    }
  }))->start();
}

// load concurrency
void Segment::load() {
  std::vector<std::future<bool>> futures;

  futures.emplace_back(std::async(std::launch::async, [=]() {
    const std::string bzip_file = localPath(log_path_).toStdString();
    const std::string decompressed_file = bzip_file + "_decompressed";
    bool is_bzip = !util::file_exists(decompressed_file);
    log = std::make_unique<LogReader>();
    return log->load(is_bzip ? bzip_file : decompressed_file, is_bzip);
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
  loaded_ = (success_cnt == futures.size());
  emit loadFinished();
}

QString Segment::localPath(const QUrl &url) {
  if (url.isLocalFile() || QFile(url.toString()).exists()) return url.toString();

  QByteArray url_no_query = url.toString(QUrl::RemoveQuery).toUtf8();
  return CACHE_DIR.filePath(QString(QCryptographicHash::hash(url_no_query, QCryptographicHash::Sha256).toHex()));
}
