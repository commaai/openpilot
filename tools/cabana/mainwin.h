#pragma once

#include <QComboBox>
#include <QJsonDocument>
#include <QMainWindow>
#include <QProgressBar>
#include <QSplitter>
#include <QStatusBar>

#include "tools/cabana/chartswidget.h"
#include "tools/cabana/detailwidget.h"
#include "tools/cabana/messageswidget.h"
#include "tools/cabana/videowidget.h"
#include "tools/cabana/tools/findsimilarbits.h"

class MainWindow : public QMainWindow {
  Q_OBJECT

public:
  MainWindow();
  void dockCharts(bool dock);
  void showStatusMessage(const QString &msg, int timeout = 0) { statusBar()->showMessage(msg, timeout); }

public slots:
  void loadDBCFromName(const QString &name);
  void loadDBCFromFingerprint();
  void loadDBCFromFile();
  void loadDBCFromClipboard();
  void saveDBCToFile();
  void saveDBCToClipboard();

signals:
  void showMessage(const QString &msg, int timeout);
  void updateProgressBar(uint64_t cur, uint64_t total, bool success);

protected:
  void createActions();
  QComboBox *createDBCSelector();
  void createStatusBar();
  void createShortcuts();
  void closeEvent(QCloseEvent *event) override;
  void DBCFileChanged();
  void updateDownloadProgress(uint64_t cur, uint64_t total, bool success);
  void setOption();
  void findSimilarBits();

  VideoWidget *video_widget;
  MessagesWidget *messages_widget;
  DetailWidget *detail_widget;
  ChartsWidget *charts_widget;
  QSplitter *splitter;
  QWidget *floating_window = nullptr;
  QVBoxLayout *r_layout;
  QProgressBar *progress_bar;
  QLabel *fingerprint_label;
  QJsonDocument fingerprint_to_dbc;
  QComboBox *dbc_combo;
};
