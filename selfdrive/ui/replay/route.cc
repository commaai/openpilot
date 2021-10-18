#include "selfdrive/ui/replay/route.h"

#include <QEventLoop>
#include <QLocale>
#include <QJsonArray>
#include <QJsonDocument>
#include <QRegExp>

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
  start_ts_ = millis_since_boot();
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

void Segment::download_callback(void *param, size_t cur_written, size_t total_size) {
  auto [s, id] = *(static_cast<std::pair<Segment *, int> *>(param));
  double current_ts = millis_since_boot();
  std::unique_lock lk(s->lock);

  s->download_written_ += cur_written;
  s->remote_file_size[id] = total_size;
  if (s->last_print_ == 0) {
    s->last_print_ = current_ts;
  } else if ((current_ts - s->last_print_) > 5 * 1000) {
    size_t total_size = 0;
    for (auto [n, size] : s->remote_file_size) {
      if (size == 0) return;

      total_size += size;
    }
    int avg_speed = s->download_written_ / ((current_ts - s->start_ts_) / 1000);
    QString percent = QString("%1%").arg((int)((s->download_written_ / (double)total_size) * 100));
    QString eta = avg_speed > 0 ? QString("%1 s").arg((total_size - s->download_written_) / avg_speed) : "--";
    qDebug() << "downloading segment" << qPrintable(QString("[%1]").arg(s->seg_num)) << ":"
             << qPrintable(QLocale().formattedDataSize(total_size)) << qPrintable(percent) << "eta:" << qPrintable(eta);
    s->last_print_ = current_ts;
  }
}

void Segment::loadFile(int id, const std::string file) {
  const bool is_remote = file.find("https://") == 0;
  const std::string local_file = is_remote ? cacheFilePath(file) : file;
  bool file_ready = util::file_exists(local_file);

  if (!file_ready && is_remote) {
    {
      std::unique_lock lk(lock);
      remote_file_size[id] = 0;
    }
    int retries = 0;
    std::pair<Segment *, int> pair {this, id};
    while (!aborting_) {
      file_ready = httpMultiPartDownload(file, local_file, id < MAX_CAMERAS ? 3 : 1, &aborting_, &Segment::download_callback, &pair);
      if (file_ready || aborting_) break;

      if (++retries > max_retries_) {
        qInfo() << "download failed after retries" << max_retries_;
        break;
      }
      qInfo() << "download failed, retrying" << retries;
    }
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
