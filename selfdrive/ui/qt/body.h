#pragma once

#include <QMovie>
#include <QLabel>

#include "selfdrive/ui/ui.h"

class BodyWindow : public QLabel {
  Q_OBJECT

public:
  BodyWindow(QWidget* parent = 0);

private:
  float fuel = 1.0;
  QMovie *awake, *sleep;
  void paintEvent(QPaintEvent*) override;

private slots:
  void updateState(const UIState &s);
};
