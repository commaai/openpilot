#pragma once

#include <QDateTime>
#include <QLayout>
#include <QLayoutItem>
#include <QPainter>
#include <QSurfaceFormat>

#include "selfdrive/common/params.h"


inline QString getBrand() {
  return Params().getBool("Passive") ? "dashcam" : "openpilot";
}

inline QString getBrandVersion() {
  return getBrand() + " v" + QString::fromStdString(Params().get("Version")).left(14).trimmed();
}

void configFont(QPainter &p, const QString &family, int size, const QString &style);
void clearLayout(QLayout* layout);
QString timeAgo(const QDateTime &date);
void setQtSurfaceFormat();

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
};

inline SignalMap *signalMap() {
  static SignalMap *signalMap = new SignalMap(qApp);
  return signalMap;
}
