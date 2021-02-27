#pragma once

#include <QVBoxLayout>
#include <QWidget>

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
