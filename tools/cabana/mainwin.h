#pragma once

#include <QSplitter>

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
  void openSettingsDlg();
  void saveSession();
  void restoreSession();

  VideoWidget *video_widget;
  MessagesWidget *messages_widget;
  DetailWidget *detail_widget;
  ChartsWidget *charts_widget;
  QSplitter *splitter;
  QWidget *floating_window = nullptr;
  QVBoxLayout *r_layout;
};
