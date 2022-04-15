#pragma once

#include <QMovie>
#include <QLabel>

#include "selfdrive/ui/ui.h"

class BodyWindow : public QLabel {
  Q_OBJECT

public:
  BodyWindow(QWidget* parent = 0);

private:
  QMovie *awake, *sleep;

private slots:
  void updateState(const UIState &s);
};
