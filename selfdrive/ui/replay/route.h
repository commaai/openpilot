#pragma once

#include <QMap>
#include <QString>

struct SegmentFile {
  QString rlog;
  QString qlog;
  QString camera;
  QString dcamera;
  QString wcamera;
  QString qcamera;
};

class Route {
public:
  Route() = default;
  Route(const QString &route);
  ~Route();
  Route &operator=(const Route &r) {
    this->route_ = r.route_;
    this->segments_ = r.segments_;
    return *this;
  }
  bool load();
  bool loadFromLocal();
  bool loadFromServer();
  bool loadFromJson(const QString &json);
  bool loadSegments(const QMap<int, QMap<QString, QString>> &segment_paths);

  inline const QString &name() const { return route_; };
  inline const QMap<int, SegmentFile> &segments() const { return segments_; }

 private:
  QString route_;
  QMap<int, SegmentFile> segments_;
};
