#pragma once

#include <QNetworkReply>
#include <QStackedLayout>
#include <QWidget>

#include "api.hpp"

class DriveStats : public QWidget {
  Q_OBJECT

public:
  explicit DriveStats(QWidget* parent = 0);

private:
  QNetworkReply* reply;
  QStackedLayout* slayout;
  CommaApi* api;
  void replyFinished();

private slots:
  void refresh();
};
