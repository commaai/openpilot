#pragma once

#include <QWidget>
#include <QNetworkReply>
#include <QStackedLayout>


class DriveStats : public QWidget {
  Q_OBJECT

public:
  explicit DriveStats(QWidget *parent = 0);

private:
  QStackedLayout *slayout;
  void replyFinished(QNetworkReply *l);
};
