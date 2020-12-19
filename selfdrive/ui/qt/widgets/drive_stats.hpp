#pragma once

#include <QFrame>
#include <QWidget>
#include <QNetworkReply>


class DriveStats : public QWidget {
  Q_OBJECT

public:
  explicit DriveStats(QWidget *parent = 0);

private:
  QFrame *f;
  void replyFinished(QNetworkReply *l);
};
