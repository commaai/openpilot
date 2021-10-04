#pragma once

#include <QDir>
#include <QObject>
#include <QString>
#include <vector>

#include "selfdrive/common/util.h"
#include "selfdrive/ui/replay/framereader.h"
#include "selfdrive/ui/replay/logreader.h"

const QDir CACHE_DIR(util::getenv("COMMA_CACHE", "/tmp/comma_download_cache/").c_str());
const int connections_per_file = 3;

struct SegmentFile {
  QString rlog;
  QString qlog;
  QString road_cam;
  QString driver_cam;
  QString wide_road_cam;
  QString qcamera;
};

class Route {
public:
  Route(const QString &route);
  bool load();

  inline const QString &name() const { return route_; };
  inline int size() const { return segments_.size(); }
  inline SegmentFile &at(int n) { return segments_[n]; }

  // public for unit tests
  std::vector<SegmentFile> segments_;

protected:
  bool loadFromJson(const QString &json);
  QString route_;
};

class Segment : public QObject {
  Q_OBJECT

public:
  Segment(int n, const SegmentFile &segment_files, bool load_dcam, bool load_ecam);
  ~Segment();
  inline bool isValid() const { return valid_; };
  inline bool isLoaded() const { return loaded_; }

  std::unique_ptr<LogReader> log;
  std::unique_ptr<FrameReader> frames[MAX_CAMERAS] = {};

signals:
  void loadFinished();

protected:
  void load();
  void downloadFile(const QString &url);
  QString localPath(const QUrl &url);

  std::atomic<bool> loaded_ = false, valid_ = false;
  std::atomic<bool> aborting_ = false;
  std::atomic<int> downloading_ = 0;
  int seg_num_ = 0;
  SegmentFile files_;
  QString road_cam_path_;
  QString log_path_;
  std::vector<QThread*> download_threads_;
};
