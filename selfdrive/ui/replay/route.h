#pragma once

#include <QMap>
#include <QString>

struct SegmentFiles {
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
  inline const QMap<int, SegmentFiles> segments() const { return segments_; }
  inline int maxSegmentNum() { return segments_.lastKey(); }
  int nextSegNum(int n) const;
  int prevSegNum(int n) const;
  inline int segmentCount() { return segments_.size(); }

 private:
  QString route_;
  QMap<int, SegmentFiles> segments_;
};
