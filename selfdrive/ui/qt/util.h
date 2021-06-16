#pragma once

#include <QApplication>
#include <QDateTime>
#include <QLayout>
#include <QLayoutItem>
#include <QPainter>
#include <QSurfaceFormat>

#include "selfdrive/common/params.h"
#include "selfdrive/ui/ui.h"


inline QString getBrand() {
  return Params().getBool("Passive") ? "dashcam" : "openpilot";
}

inline QString getBrandVersion() {
  return getBrand() + " v" + QString::fromStdString(Params().get("Version")).left(14).trimmed();
}

void configFont(QPainter &p, const QString &family, int size, const QString &style);
void clearLayout(QLayout* layout);
QString timeAgo(const QDateTime &date);
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


class SignalMap : public QObject {
  Q_OBJECT

public:
  explicit SignalMap(QObject *parent) : QObject(parent) {}

signals:
  void offroadTransition(bool offroad);
  void reviewTrainingGuide();
  void openSettings();
  void closeSettings();
  void showDriverView();
  void displayPowerChanged(bool on);
  void uiUpdate(const UIState &s);

  void toggleParameter(const QString &, bool);
};

inline SignalMap *signalMap() {
  static SignalMap *signalMap = new SignalMap(qApp);
  return signalMap;
}
