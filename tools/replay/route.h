#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <QDateTime>
#include <QFutureSynchronizer>

#include "tools/replay/framereader.h"
#include "tools/replay/logreader.h"
#include "tools/replay/util.h"

enum class RouteLoadError {
  None,
  AccessDenied,
  NetworkError,
  FileNotFound,
  UnknownError
};

struct RouteIdentifier {
  std::string dongle_id;
  std::string timestamp;
  int begin_segment = 0;
  int end_segment = -1;
  std::string str;
};

struct SegmentFile {
  std::string rlog;
  std::string qlog;
  std::string road_cam;
  std::string driver_cam;
  std::string wide_road_cam;
  std::string qcamera;
};

class Route {
public:
  Route(const std::string &route, const std::string &data_dir = {});
  bool load();
  RouteLoadError lastError() const { return err_; }
  inline const std::string &name() const { return route_.str; }
  inline const QDateTime datetime() const { return date_time_; }
  inline const std::string &dir() const { return data_dir_; }
  inline const RouteIdentifier &identifier() const { return route_; }
  inline const std::map<int, SegmentFile> &segments() const { return segments_; }
  inline const SegmentFile &at(int n) { return segments_.at(n); }
  static RouteIdentifier parseRoute(const std::string &str);

protected:
  bool loadFromLocal();
  bool loadFromServer(int retries = 3);
  bool loadFromJson(const QString &json);
  void addFileToSegment(int seg_num, const std::string &file);
  RouteIdentifier route_ = {};
  std::string data_dir_;
  std::map<int, SegmentFile> segments_;
  QDateTime date_time_;
  RouteLoadError err_ = RouteLoadError::None;
};

class Segment : public QObject {
  Q_OBJECT

public:
  Segment(int n, const SegmentFile &files, uint32_t flags, const std::vector<bool> &filters = {});
  ~Segment();
  inline bool isLoaded() const { return !loading_ && !abort_; }

  const int seg_num = 0;
  std::unique_ptr<LogReader> log;
  std::unique_ptr<FrameReader> frames[MAX_CAMERAS] = {};

signals:
  void loadFinished(bool success);

protected:
  void loadFile(int id, const std::string file);

  std::atomic<bool> abort_ = false;
  std::atomic<int> loading_ = 0;
  QFutureSynchronizer<void> synchronizer_;
  uint32_t flags;
  std::vector<bool> filters_;
};
