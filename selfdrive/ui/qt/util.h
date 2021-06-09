#pragma once

#include <QtWidgets>

#include "selfdrive/common/params.h"


inline void configFont(QPainter &p, const QString &family, int size, const QString &style) {
  QFont f(family);
  f.setPixelSize(size);
  f.setStyleName(style);
  p.setFont(f);
}

inline void clearLayout(QLayout* layout) {
  while (QLayoutItem* item = layout->takeAt(0)) {
    if (QWidget* widget = item->widget()) {
      widget->deleteLater();
    }
    if (QLayout* childLayout = item->layout()) {
      clearLayout(childLayout);
    }
    delete item;
  }
}

inline QString timeAgo(const QDateTime &date) {
  int diff = date.secsTo(QDateTime::currentDateTimeUtc());

  QString s;
  if (diff < 60) {
    s = "now";
  } else if (diff < 60 * 60) {
    int minutes = diff / 60;
    s = QString("%1 minute%2 ago").arg(minutes).arg(minutes > 1 ? "s" : "");
  } else if (diff < 60 * 60 * 24) {
    int hours = diff / (60 * 60);
    s = QString("%1 hour%2 ago").arg(hours).arg(hours > 1 ? "s" : "");
  } else if (diff < 3600 * 24 * 7) {
    int days = diff / (60 * 60 * 24);
    s = QString("%1 day%2 ago").arg(days).arg(days > 1 ? "s" : "");
  } else {
    s = date.date().toString();
  }

  return s;
}

inline void setQtSurfaceFormat() {
  QSurfaceFormat fmt;
#ifdef __APPLE__
  fmt.setVersion(3, 2);
  fmt.setProfile(QSurfaceFormat::OpenGLContextProfile::CoreProfile);
  fmt.setRenderableType(QSurfaceFormat::OpenGL);
#else
  fmt.setRenderableType(QSurfaceFormat::OpenGLES);
#endif
  QSurfaceFormat::setDefaultFormat(fmt);
}

class QParams : public Params {
  public:
  QParams(bool persistent_param = false) : Params(persistent_param) {}
  QParams(const QString &path) : Params(path.toStdString()) {}
  inline QString Get(const QString &key, bool block = false) {
    return QString::fromStdString(Params::get(qPrintable(key), block));
  }
  inline QString Get(const char *key, bool block = false) {
    return QString::fromStdString(Params::get(key, block));
  }
  inline int Put(const char *key, const QString &val) {
    return Params::put(key, val.toStdString());
  }
  inline int Put(const QString &key, const QString &val) {
    return Put(qPrintable(key), val);
  }
};

static inline QParams qParams;

inline QString getBrand() {
  return qParams.getBool("Passive") ? "dashcam" : "openpilot";
}

inline QString getBrandVersion() {
  return getBrand() + " v" + qParams.Get("Version").left(14).trimmed();
}
