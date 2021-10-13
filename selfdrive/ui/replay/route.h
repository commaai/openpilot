#pragma once

#include <QDir>
#include <QFutureSynchronizer>

#include "selfdrive/common/util.h"
#include "selfdrive/ui/replay/framereader.h"
#include "selfdrive/ui/replay/logreader.h"

const QDir CACHE_DIR(util::getenv("COMMA_CACHE", "/tmp/comma_download_cache/").c_str());

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
  Route(const QString &route, const QString &data_dir = {}) : route_(route), data_dir_(data_dir) {};
  bool load();
  inline const QString &name() const { return route_; };
  inline const std::map<int, SegmentFile> &segments() const { return segments_; }
  inline const SegmentFile &at(int n) { return segments_.at(n); }

protected:
  bool loadFromLocal();
  bool loadFromServer();
  bool loadFromJson(const QString &json);
  void addFileToSegment(int seg_num, const QString &file);
  QString route_;
  QString data_dir_;
  std::map<int, SegmentFile> segments_;
};

class Segment : public QObject {
  Q_OBJECT

public:
  Segment(int n, const SegmentFile &files, bool load_dcam, bool load_ecam);
  ~Segment();
  inline bool isLoaded() const { return loading_ == 0; }

  std::unique_ptr<LogReader> log;
  std::unique_ptr<FrameReader> frames[MAX_CAMERAS] = {};

signals:
  void loadFinished();

protected:
  void loadFile(int id, const std::string file);
  std::string cacheFilePath(const std::string &file);

  std::atomic<bool> aborting_ = false;
  std::atomic<int> loading_ = 0;
  int seg_num_ = 0;
  QFutureSynchronizer<void> synchronizer_;
};
