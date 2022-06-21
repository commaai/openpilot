#pragma once

#include <QJsonDocument>
#include <QLabel>

class DriveStats : public QFrame {
  Q_OBJECT

public:
  explicit DriveStats(QWidget* parent = 0);

private:
  void showEvent(QShowEvent *event) override;
  void updateStats();
  inline QString getDistanceUnit() const { return metric_ ? tr("KM") : tr("Miles"); }

  bool metric_;
  QJsonDocument stats_;
  struct StatsLabels {
    QLabel *routes, *distance, *distance_unit, *hours;
  } all_, week_;

private slots:
  void parseResponse(const QString &response, bool success);
};
