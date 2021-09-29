#include "selfdrive/ui/replay/route.h"

#include <curl/curl.h>

#include <QDir>
#include <QEventLoop>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QRegExp>
#include <QThread>
#include <future>

#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/qt/api.h"

struct CURLGlobalInitializer {
  CURLGlobalInitializer() { curl_global_init(CURL_GLOBAL_DEFAULT); }
  ~CURLGlobalInitializer() { curl_global_cleanup(); }
};

struct MultiPartWriter {
  int64_t offset;
  int64_t end;
  FILE *fp;
};

static size_t write_cb(char *data, size_t n, size_t l, void *userp) {
  MultiPartWriter *w = (MultiPartWriter *)userp;
  fseek(w->fp, w->offset, SEEK_SET);
  fwrite(data, l, n, w->fp);
  w->offset += n * l;
  return n * l;
}

static size_t dumy_write_cb(char *data, size_t n, size_t l, void *userp) { return n * l; }

int64_t getDownloadContentLength(const std::string &url) {
  CURL *curl = curl_easy_init();
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, dumy_write_cb);
  curl_easy_setopt(curl, CURLOPT_HEADER, 1);
  curl_easy_setopt(curl, CURLOPT_NOBODY, 1);
  CURLcode res = curl_easy_perform(curl);
  double content_length = -1;
  if (res == CURLE_OK) {
    res = curl_easy_getinfo(curl, CURLINFO_CONTENT_LENGTH_DOWNLOAD, &content_length);
  }
  curl_easy_cleanup(curl);
  return res == CURLE_OK ? (int64_t)content_length : -1;
}

bool httpMultiPartDownload(const std::string &url, const std::string &target_file, int parts, std::atomic<bool> *abort) {
  int64_t content_length = getDownloadContentLength(url);
  if (content_length == -1) return false;

  std::string tmp_file = target_file + ".tmp";
  FILE *fp = fopen(tmp_file.c_str(), "wb");
  // create a sparse file
  fseek(fp, content_length, SEEK_SET);

  CURLM *cm = curl_multi_init();
  std::map<CURL *, MultiPartWriter> writers;
  const int part_size = content_length / parts;
  for (int i = 0; i < parts; ++i) {
    CURL *eh = curl_easy_init();
    writers[eh] = {
        .fp = fp,
        .offset = i * part_size,
        .end = i == parts - 1 ? content_length - 1 : (i + 1) * part_size - 1,
    };
    curl_easy_setopt(eh, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(eh, CURLOPT_WRITEDATA, (void *)(&writers[eh]));
    curl_easy_setopt(eh, CURLOPT_URL, url.c_str());
    curl_easy_setopt(eh, CURLOPT_RANGE, util::string_format("%d-%d", writers[eh].offset, writers[eh].end).c_str());
    curl_easy_setopt(eh, CURLOPT_HTTPGET, 1);
    curl_easy_setopt(eh, CURLOPT_NOSIGNAL, 1);
    curl_easy_setopt(eh, CURLOPT_FOLLOWLOCATION, 1);
    curl_multi_add_handle(cm, eh);
  }

  int running = 1, success_cnt = 0;
  while (!(abort && abort->load())){
    CURLMcode ret = curl_multi_perform(cm, &running);

    if (!running) {
      CURLMsg *msg;
      int msgs_left = -1;
      while ((msg = curl_multi_info_read(cm, &msgs_left))) {
        if (msg->msg == CURLMSG_DONE && msg->data.result == CURLE_OK) {
          int http_status_code = 0;
          curl_easy_getinfo(msg->easy_handle, CURLINFO_RESPONSE_CODE, &http_status_code);
          success_cnt += (http_status_code == 206);
        }
      }
      break;
    }

    if (ret == CURLM_OK) {
      curl_multi_wait(cm, nullptr, 0, 1000, nullptr);
    }
  };

  fclose(fp);
  bool success = success_cnt == parts;
  if (success) {
    success = ::rename(tmp_file.c_str(), target_file.c_str()) == 0;
  }

  // cleanup curl
  for (auto &[e, w] : writers) {
    curl_multi_remove_handle(cm, e);
    curl_easy_cleanup(e);
  }
  curl_multi_cleanup(cm);
  return success;
}

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

Segment::Segment(int n, const SegmentFile &segment_files, bool load_dcam, bool load_ecam) : seg_num_(n), files_(segment_files) {
  static CURLGlobalInitializer curl_initializer;
  static std::once_flag once_flag;
  std::call_once(once_flag, [=]() {
    if (!QDir(CACHE_DIR).exists()) QDir().mkdir(CACHE_DIR);
  });

  // fallback to qcamera/qlog
  road_cam_path_ = files_.road_cam.isEmpty() ? files_.qcamera : files_.road_cam;
  log_path_ = files_.rlog.isEmpty() ? files_.qlog : files_.rlog;

  valid_ = !log_path_.isEmpty() && !road_cam_path_.isEmpty();
  if (!valid_) return;

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
  }
}

Segment::~Segment() {
  aborting_ = true;
  for (auto &t : download_threads_) {
    if (t->isRunning()) t->wait();
  }
}

void Segment::downloadFile(const QString &url) {
  qDebug() << "download" << url;
  download_threads_.emplace_back(QThread::create([=]() {
    httpMultiPartDownload(url.toStdString(), localPath(url).toStdString(), connections_per_file, &aborting_);
    if (--downloading_ == 0 && !aborting_) {
      load();
    }
  }))->start();
}

// load concurrency
void Segment::load() {
  std::vector<std::future<bool>> futures;
  futures.emplace_back(std::async(std::launch::async, [=]() {
    log = std::make_unique<LogReader>();
    return log->load(localPath(log_path_).toStdString());
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
