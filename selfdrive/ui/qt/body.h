#pragma once

#include <QStackedLayout>
#include <QWidget>

#include "selfdrive/ui/ui.h"


class BodyWindow : public QWidget {
  Q_OBJECT

public:
  BodyWindow(QWidget* parent = 0);

private:
  void paintEvent(QPaintEvent *event);

private slots:
  void updateState(const UIState &s);
};
