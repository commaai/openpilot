#pragma once

#include <QDateTime>
#include <QLayout>
#include <QMouseEvent>
#include <QPainter>
#include <QSurfaceFormat>
#include <QWidget>

#include "selfdrive/common/params.h"


void configFont(QPainter &p, const QString &family, int size, const QString &style);
void clearLayout(QLayout* layout);
QString timeAgo(const QDateTime &date);

inline QString getBrand() {
  return Params().getBool("Passive") ? "dashcam" : "openpilot";
}

inline QString getBrandVersion() {
  return getBrand() + " v" + QString::fromStdString(Params().get("Version")).left(14).trimmed();
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

class ClickableWidget : public QWidget
{
  Q_OBJECT

public:
  ClickableWidget(QWidget *parent = nullptr);

protected:
  void mouseReleaseEvent(QMouseEvent *event) override;
  void paintEvent(QPaintEvent *) override;

signals:
  void clicked();
};
