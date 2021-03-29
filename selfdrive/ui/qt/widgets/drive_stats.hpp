#pragma once

#include <QLabel>
#include <QNetworkReply>
class DriveStats : public QWidget {
  Q_OBJECT

public:
  explicit DriveStats(QWidget* parent = 0);

private:
  struct StatsLabels {
    QLabel *routes, *distance, *hours;
  } all_, week_;

private slots:
  void parseResponse(QNetworkReply::NetworkError err, const QString& response);
};
