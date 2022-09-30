#pragma once

#include <QWidget>

#include "tools/cabana/canwidget.h"
#include "tools/cabana/videowidget.h"

class MainWindow : public QWidget {
Q_OBJECT

public:
  MainWindow();

  VideoWidget *video_widget;
  CanWidget *can_widget;
};
