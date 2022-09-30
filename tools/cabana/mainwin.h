#pragma once

#include <QWidget>

#include "tools/cabana/canwidget.h"
#include "tools/cabana/chartswidget.h"
#include "tools/cabana/detailwidget.h"
#include "tools/cabana/videowidget.h"
#include "tools/cabana/parser.h"

class MainWindow : public QWidget {
Q_OBJECT

public:
  MainWindow();
public slots:
  void updated();
protected:

  VideoWidget *video_widget;
  CanWidget *can_widget;
  DetailWidget *detail_widget;
  ChartsWidget *charts_widget;
};
