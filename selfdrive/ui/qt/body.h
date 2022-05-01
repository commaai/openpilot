#pragma once

#include <QGridLayout>
#include <QLabel>
#include <QPainter>

#include "selfdrive/ui/ui.h"

class BodyWindow : public QWidget {
  Q_OBJECT

public:
  BodyWindow(QWidget* parent = 0);

private:
  QGridLayout *layout;
  QLabel *battery;

  int batteryDots;

  QString generateBatteryText(float fuelGauge);

protected:
  void paintEvent(QPaintEvent *event);

private slots:
  void updateState(const UIState &s);
};
