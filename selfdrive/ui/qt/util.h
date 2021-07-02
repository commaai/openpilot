#pragma once

#include <QDateTime>
#include <QLayout>
#include <QMouseEvent>
#include <QPainter>
#include <QSurfaceFormat>
#include <QWidget>

QString getBrand();
QString getBrandVersion();
void configFont(QPainter &p, const QString &family, int size, const QString &style);
void clearLayout(QLayout* layout);
void setQtSurfaceFormat();
QString timeAgo(const QDateTime &date);
void swagLogMessageHandler(QtMsgType type, const QMessageLogContext &context, const QString &msg);

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
