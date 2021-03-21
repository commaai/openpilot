#pragma once

#include <QLabel>

class DriveStats : public QWidget {
  Q_OBJECT

public:
  explicit DriveStats(QWidget* parent = 0);

private:
  void showEvent(QShowEvent *event) override {
    
  }

  struct StatsLabels {
    QLabel *routes, *distance, *hours;
  } all_, week_;

private slots:
  void parseResponse(QString response, bool save = true);
};
