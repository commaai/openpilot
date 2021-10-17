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
  inline bool isLoaded() const { return !loading_ && success_; }

  const int seg_num = 0;
  std::unique_ptr<LogReader> log;
  std::unique_ptr<FrameReader> frames[MAX_CAMERAS] = {};

signals:
  void loadFinished(bool success);

protected:
  void loadFile(int id, const std::string file);
  std::string cacheFilePath(const std::string &file);

  std::atomic<bool> success_ = true, aborting_ = false;
  std::atomic<int> loading_ = 0;
  QFutureSynchronizer<void> synchronizer_;
};

class SegmentManager : public QObject {
  Q_OBJECT

public:
  SegmentManager(int cache_segment_cnt = 8, QObject *parent = nullptr) : cache_size_(cache_segment_cnt), QObject(parent) {}
  std::shared_ptr<Segment> get(int n, const SegmentFile &files, bool load_cam, bool load_eccam);
  inline int getCacheSize() const { return cache_size_; }
  inline void setCacheSize(int size) { cache_size_ = size; }

signals:
  void segmentLoadFinished(int n, bool success);

protected:
  struct SegmentData {
    std::shared_ptr<Segment> segment;
    double last_used;
  };
  std::map<int, std::unique_ptr<SegmentData>> segments_;
  int cache_size_;
};
