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
#include <set>

#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/qt/api.h"

namespace {

QString local_path(const QUrl &url) {
  QByteArray url_no_query = url.toString(QUrl::RemoveQuery).toUtf8();
  return CACHE_DIR + QString(QCryptographicHash::hash(url_no_query, QCryptographicHash::Sha256).toHex());
}
struct MultiPartWriter {
  int offset;
  int end;
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

int32_t getUrlContentLength(const std::string &url) {
  CURL *curl = curl_easy_init();
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, dumy_write_cb);
  curl_easy_setopt(curl, CURLOPT_HEADER, 1);
  curl_easy_setopt(curl, CURLOPT_NOBODY, 1);
  int ret = curl_easy_perform(curl);
  double content_length = 0;
  curl_easy_getinfo(curl, CURLINFO_CONTENT_LENGTH_DOWNLOAD, &content_length);
  curl_easy_cleanup(curl);
  return ret == CURLE_OK ? (int32_t)content_length : -1;
}

void httpMultiPartDownload(const std::string &url, int parts, bool *abort) {
  int content_length = getUrlContentLength(url);
  if (content_length == -1) return;

  std::string cache_file_path = local_path(QString::fromStdString(url)).toStdString();
  std::string tmp_cache_file_path = cache_file_path + ".tmp";
  FILE *fp = fopen(tmp_cache_file_path.c_str(), "wb");
  // create a sparse file
  fseek(fp, (long)content_length, SEEK_SET);
  fputc('\0', fp);

  CURLM *cm = curl_multi_init();
  std::unique_ptr<MultiPartWriter[]> writers = std::make_unique<MultiPartWriter[]>(parts);
  std::set<CURL *> easy_handles;
  const int part_size = content_length / parts;
  
  for (int i = 0; i < parts; ++i) {
    writers[i].fp = fp;
    writers[i].offset = i * part_size;
    writers[i].end = i == parts - 1 ? content_length - 1 : writers[i].offset + part_size - 1;

    CURL *eh = curl_easy_init();
    curl_easy_setopt(eh, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(eh, CURLOPT_WRITEDATA, (void *)(&writers[i]));
    curl_easy_setopt(eh, CURLOPT_URL, url.c_str());
    curl_easy_setopt(eh, CURLOPT_RANGE, util::string_format("%d-%d", writers[i].offset, writers[i].end).c_str());
    curl_easy_setopt(eh, CURLOPT_HTTPGET, 1);
    curl_easy_setopt(eh, CURLOPT_NOSIGNAL, 1);
    curl_easy_setopt(eh, CURLOPT_FOLLOWLOCATION, 1);
    curl_multi_add_handle(cm, eh);
    easy_handles.insert(eh);
  }

  int msgs_left = -1, still_alive = 1;
  while (!(*abort)) {
    curl_multi_perform(cm, &still_alive);
    if (!still_alive) break;

    CURLMsg *msg;
    while ((msg = curl_multi_info_read(cm, &msgs_left))) {
      if (msg->msg == CURLMSG_DONE) {
        curl_multi_remove_handle(cm, msg->easy_handle);
        curl_easy_cleanup(msg->easy_handle);
        easy_handles.erase(msg->easy_handle);
      } else {
        qDebug() << "faild download" << url.c_str() << msg->msg;
      }
    }
    curl_multi_wait(cm, NULL, 0, 100, NULL);
  }

  fclose(fp);
  if (!*abort) {
    ::rename(tmp_cache_file_path.c_str(), cache_file_path.c_str());
  }
  for (auto &e : easy_handles) {
    curl_multi_remove_handle(cm, e);
    curl_easy_cleanup(e);
  }
  curl_multi_cleanup(cm);
}

}  // namespace

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
  static std::once_flag once_flag;
  std::call_once(once_flag, [=]() {
    if (!QDir(CACHE_DIR).exists()) QDir().mkdir(CACHE_DIR);
  });

  // fallback to qcamera
  road_cam_path_ = files_.road_cam.isEmpty() ? files_.qcamera : files_.road_cam;
  valid_ = !files_.rlog.isEmpty() && !road_cam_path_.isEmpty();
  if (!valid_) return;

  if (!load_dcam) {
    files_.driver_cam = "";
  }
  if (!load_ecam) {
    files_.wide_road_cam = "";
  }

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
  aborting_ = true;
  for (auto &t : download_threads_) {
    if (t->isRunning()) t->wait();
  }
}

void Segment::downloadFile(const QString &url) {
  QThread *thread = QThread::create([=]() {
    httpMultiPartDownload(url.toStdString(), 3, &aborting_);
    if (--downloading_ == 0 && !aborting_) {
      load();
    }
  });
  download_threads_.push_back(thread);
  thread->start();
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
  return local_path(url);
}
