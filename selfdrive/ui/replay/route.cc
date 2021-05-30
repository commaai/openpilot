#include "selfdrive/ui/replay/route.h"

#include <regex>

#include <QDebug>
#include <QDir>
#include <QEventLoop>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

#include "selfdrive/common/util.h"
#include "selfdrive/hardware/hw.h"
#include "selfdrive/ui/qt/api.h"

const std::string LOG_ROOT =
    Hardware::PC() ? util::getenv_default("HOME", "/.comma/media/0/realdata", "/data/media/0/realdata")
                   : "/data/media/0/realdata";

Route::Route(const QString &route) : route_(route) {}

Route::~Route() {}

bool Route::load() {
  if (!loadFromLocal()) {
    return loadFromServer();
  }
  return true;
}

bool Route::loadFromServer() {
  bool ret = false;
  const QString url = "https://api.commadotai.com/v1/route/" + route_ + "/files";

  QEventLoop loop;
  auto onError = [&loop](const QString &err) {
    qInfo() << err;
    loop.quit();
  };
  HttpRequest http(nullptr, url, "", !Hardware::PC());
  QObject::connect(&http, &HttpRequest::failedResponse, onError);
  QObject::connect(&http, &HttpRequest::timeoutResponse, onError);
  QObject::connect(&http, &HttpRequest::receivedResponse, [&](const QString json) {
    ret = loadFromJson(json);
    loop.quit();
  });
  loop.exec();
  return ret;
}

bool Route::loadFromLocal() {
  QStringList list = route_.split('|');
  if (list.size() != 2) return false;

  QDir log_root(LOG_ROOT.c_str());
  QStringList folders = log_root.entryList(QStringList() << list[1] + "*", QDir::Dirs | QDir::NoDot, QDir::NoSort);
  if (folders.isEmpty()) return false;

  QMap<int, QMap<QString, QString>> segment_paths;
  for (auto folder : folders) {
    const int seg_num = folder.split("--")[2].toInt();
    auto &paths = segment_paths[seg_num];

    QDir segment(log_root.filePath(folder));
    for (auto f : segment.entryList(QDir::Files)) {
      const QString file_path = "file://" + segment.filePath(f);
      if (f.startsWith("fcamera")) {
        paths["cameras"] = file_path;
      } else if (f.startsWith("dcamera")) {
        paths["dcameras"] = file_path;
      } else if (f.startsWith("ecamera")) {
        paths["ecameras"] = file_path;
      } else if (f.startsWith("qcamera")) {
        paths["qcameras"] = file_path;
      } else if (f.startsWith("rlog")) {
        paths["logs"] = file_path;
      } else if (f.startsWith("qlog")) {
        paths["qlogs"] = file_path;
      }
    }
  }
  return loadSegments(segment_paths);
}

bool Route::loadFromJson(const QString &json) {
  QJsonDocument doc = QJsonDocument::fromJson(json.trimmed().toUtf8());
  if (doc.isNull()) {
    qInfo() << "JSON Parse failed";
    return false;
  }
  std::regex regexp(R"(^(.*?)\/(\d+)\/(.*?))");
  QMap<int, QMap<QString, QString>> segment_paths;

  QJsonObject obj = doc.object();
  for (const QString &key : obj.keys()) {
    for (const auto &p : obj[key].toArray()) {
      std::string path = p.toString().toStdString();
      if (std::smatch match; std::regex_match(path, match, regexp)) {
        const int seg_num = std::stoi(match[2].str());
        segment_paths[seg_num][key] = p.toString();
      }
    }
  }
  return loadSegments(segment_paths);
}

bool Route::loadSegments(const QMap<int, QMap<QString, QString>> &segment_paths) {
  segments_.clear();

  for (int seg_num : segment_paths.keys()) {
    auto &paths = segment_paths[seg_num];
    SegmentFiles &files = segments_[seg_num];
    files.rlog = paths.value("logs");
    files.qlog = paths.value("qlogs");
    files.camera = paths.value("cameras");
    files.dcamera = paths.value("dcameras");
    files.wcamera = paths.value("ecameras");
    files.qcamera = paths.value("qcameras");
  }
  return true;
}

int Route::nextSegNum(int n) const {
  auto it = segments_.upperBound(n);
  return it != segments_.end() ? it.key() : -1;
}

int Route::prevSegNum(int n) const {
  auto it = segments_.lowerBound(n);
  return it != segments_.end() ? it.key() : -1;
}
