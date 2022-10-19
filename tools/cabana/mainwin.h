#pragma once

#include "tools/cabana/chartswidget.h"
#include "tools/cabana/detailwidget.h"
#include "tools/cabana/messageswidget.h"
#include "tools/cabana/videowidget.h"

class MainWindow : public QWidget {
  Q_OBJECT

public:
  MainWindow();
  void dockCharts(bool dock);

protected:
  void closeEvent(QCloseEvent *event) override;
  void setOption();

  VideoWidget *video_widget;
  MessagesWidget *messages_widget;
  DetailWidget *detail_widget;
  ChartsWidget *charts_widget;
  QWidget *floating_window = nullptr;
  QVBoxLayout *r_layout;
};

class SettingsDlg : public QDialog {
  Q_OBJECT

public:
  SettingsDlg(QWidget *parent);
  void save();
  QSpinBox *fps;
  QSpinBox *log_size ;
  QSpinBox *cached_segment;
  QSpinBox *chart_height;
};
