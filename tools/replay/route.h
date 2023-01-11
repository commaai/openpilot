#pragma once

#include <QDateTime>
#include <QFutureSynchronizer>

#include "tools/replay/framereader.h"
#include "tools/replay/logreader.h"
#include "tools/replay/util.h"

struct RouteIdentifier {
  QString dongle_id;
  QString timestamp;
  int segment_id;
  QString str;
};

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
  Route(const QString &route, const QString &data_dir = {});
  bool load();
  inline const QString &name() const { return route_.str; }
  inline const QDateTime datetime() const { return date_time_; }
  inline const QString &dir() const { return data_dir_; }
  inline const RouteIdentifier &identifier() const { return route_; }
  inline const std::map<int, SegmentFile> &segments() const { return segments_; }
  inline const SegmentFile &at(int n) { return segments_.at(n); }
  static RouteIdentifier parseRoute(const QString &str);

protected:
  bool loadFromLocal();
  bool loadFromServer();
  bool loadFromJson(const QString &json);
  void addFileToSegment(int seg_num, const QString &file);
  RouteIdentifier route_ = {};
  QString data_dir_;
  std::map<int, SegmentFile> segments_;
  QDateTime date_time_;
};

class Segment : public QObject {
  Q_OBJECT

public:
  Segment(int n, const SegmentFile &files, uint32_t flags, const std::set<cereal::Event::Which> &allow = {});
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
  std::set<cereal::Event::Which> allow;
};
