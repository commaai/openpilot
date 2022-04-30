#pragma once

#include <QVBoxLayout>
#include <QMovie>
#include <QLabel>

#include "selfdrive/ui/ui.h"

class BodyWindow : public QWidget {
  Q_OBJECT

public:
  BodyWindow(QWidget* parent = 0);

private:
  QVBoxLayout *layout;
  QLabel *face;
  QMovie *awake, *sleep;
  QLabel *battery;

private slots:
  void updateState(const UIState &s);
};
