#pragma once

#include <QNetworkReply>
#include <QVBoxLayout>
#include <QWidget>

#include "api.hpp"

class DriveStats : public QWidget {
  Q_OBJECT

public:
  explicit DriveStats(QWidget* parent = 0);

private:
  QVBoxLayout* vlayout;

private slots:
  void parseError(QString response);
  void parseResponse(QString response);
};
