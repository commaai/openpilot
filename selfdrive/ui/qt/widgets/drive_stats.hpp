#pragma once

#include <QWidget>
#include <QNetworkReply>
#include <QDateTime>

class DriveStats : public QWidget {
  Q_OBJECT

public:
  explicit DriveStats(QWidget *parent = 0);

private:
  void replyFinished(QNetworkReply *l);
};
