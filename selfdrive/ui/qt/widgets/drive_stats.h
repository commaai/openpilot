#pragma once

#include <QJsonDocument>
#include <QLabel>

#include "selfdrive/ui/qt/widgets/controls.h"

class DriveStats : public QFrame, public UI {
  Q_OBJECT

public:
  explicit DriveStats(QWidget* parent = 0);
  void retranslateUi() override;

private:
  void showEvent(QShowEvent *event) override;
  void updateStats();
  inline QString getDistanceUnit() const { return metric_ ? tr("KM") : tr("Miles"); }

  bool metric_;
  QJsonDocument stats_;
  struct StatsLabels {
    QLabel *title;
    QLabel *routes_title, *distance_title, *hours_title;
    QLabel *routes, *distance, *hours;
  } all_, week_;

  QLabel *all_time, *past_week;

private slots:
  void parseResponse(const QString &response, bool success);
};
