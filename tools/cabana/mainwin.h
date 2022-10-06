#pragma once

#include <QPushButton>
#include <QWidget>

#include "tools/cabana/chartswidget.h"
#include "tools/cabana/detailwidget.h"
#include "tools/cabana/messageswidget.h"
#include "tools/cabana/parser.h"
#include "tools/cabana/videowidget.h"

class FloatWindow : public QWidget {
  Q_OBJECT

public:
  FloatWindow(QWidget *parent = nullptr);
  void closeEvent(QCloseEvent *event) override;

signals:
  void closing();
};

class MainWindow : public QWidget {
  Q_OBJECT

public:
  MainWindow();
  void floatingCharts(bool floating);

protected:
  void closeEvent(QCloseEvent *event) override;

  VideoWidget *video_widget;
  MessagesWidget *messages_widget;
  DetailWidget *detail_widget;
  ChartsWidget *charts_widget;
  FloatWindow *floating_window = nullptr;
  QVBoxLayout *r_layout;
};
