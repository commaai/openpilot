#pragma once

#include <QMovie>
#include <QLabel>

#include "selfdrive/common/util.h"
#include "selfdrive/ui/ui.h"

class BodyWindow : public QLabel {
  Q_OBJECT

public:
  BodyWindow(QWidget* parent = 0);

private:
  bool charging = false;
  FirstOrderFilter fuel_filter;
  QMovie *awake, *sleep;
  void paintEvent(QPaintEvent*) override;

private slots:
  void updateState(const UIState &s);
};
