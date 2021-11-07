#pragma once

#include <optional>

#include <QDateTime>
#include <QLayout>
#include <QMouseEvent>
#include <QPainter>
#include <QSurfaceFormat>
#include <QWidget>

const double MILE_TO_KM = 1.609344;
const double KM_TO_MILE = 1. / MILE_TO_KM;
const double MS_TO_KPH = 3.6;
const double MS_TO_MPH = MS_TO_KPH * KM_TO_MILE;
const double METER_2_MILE = KM_TO_MILE / 1000.0;
const double METER_2_FOOT = 3.28084;

QString getBrand();
QString getBrandVersion();
std::optional<QString> getDongleId();
void configFont(QPainter &p, const QString &family, int size, const QString &style);
void clearLayout(QLayout* layout);
void setQtSurfaceFormat();
QString timeAgo(const QDateTime &date);
void swagLogMessageHandler(QtMsgType type, const QMessageLogContext &context, const QString &msg);
void initApp();
QWidget* topWidget (QWidget* widget);


// convenience class for wrapping layouts
class LayoutWidget : public QWidget {
  Q_OBJECT

public:
  LayoutWidget(QLayout *l, QWidget *parent = nullptr) : QWidget(parent) {
    setLayout(l);
  };
};

class ClickableWidget : public QWidget {
  Q_OBJECT

public:
  ClickableWidget(QWidget *parent = nullptr);

protected:
  void mouseReleaseEvent(QMouseEvent *event) override;
  void paintEvent(QPaintEvent *) override;

signals:
  void clicked();
};
